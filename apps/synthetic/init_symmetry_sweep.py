"""Init-symmetry sweep: constant raw offsets on the TNBBeta head, pilot configs.

Usage: init_symmetry_sweep.py out.jsonl FAMILIES LRS DRAWS CONDS [SHARD NUM_SHARDS]
(comma lists; LRS may contain 'tuned'). Resumable: cells already in out.jsonl are
skipped. Device from select_device() (exit 75 without GPU). Exploratory, uncommitted;
see writeup/results/tnb_init_symmetry_and_head_offset_2026-11-01.md.
Offsets are added to the raw head output inside GraphVAE._posterior (a fixed shift,
like initial_kappa), via monkeypatch only; src/ is untouched.
"""

import json
import math
import os
from pathlib import Path
import sys
import time

import torch

from apps.link_prediction.main import run_once
from apps.synthetic import graph_search as gs
from tnbbeta_vae.data.graph_space import GraphSpec, sample_graph
from tnbbeta_vae.data.planetoid import Graph, split_edges
from tnbbeta_vae.models import GraphVAE
from tnbbeta_vae.training import select_device

torch.set_num_threads(1)


def logit(x):
    return math.log(x / (1 - x))


def spinv(x):
    return math.log(math.expm1(x - 1e-4))


def tgt(p=None, q=None, eps=None, ws=1.0):
    return dict(
        p=None if p is None else logit(p),
        q=None if q is None else logit(q),
        eps=None if eps is None else spinv(eps),
        ws=ws,
        targets=dict(p=p, q=q, eps=eps),
    )


CONDS = {  # base condition, shift
    "default": ("tnbbeta", tgt()),
    "p0.6": ("tnbbeta", tgt(p=0.6)),
    "p0.75": ("tnbbeta", tgt(p=0.75)),
    "p0.9": ("tnbbeta", tgt(p=0.9)),
    "eps2": ("tnbbeta", tgt(eps=2.0)),
    "eps8": ("tnbbeta", tgt(eps=8.0)),
    "eps8.5": ("tnbbeta", tgt(eps=8.5)),
    "q0.1": ("tnbbeta", tgt(q=0.1)),
    "q0.9": ("tnbbeta", tgt(q=0.9)),
    "w10": ("tnbbeta", tgt(ws=10.0)),
    "w30": ("tnbbeta", tgt(ws=30.0)),
    "p0.58": ("tnbbeta", tgt(p=0.58)),  # Ew ~0.105 = vMF at softplus(0)+1, d=16
    "p0.75q0.25": ("tnbbeta", tgt(p=0.75, q=0.25)),
    "p0.75eps2": ("tnbbeta", tgt(p=0.75, eps=2.0)),
    "p0.75eps8.5": ("tnbbeta", tgt(p=0.75, eps=8.5)),
    "vmf_init": ("vmf_init", None),
    "vmf_default": ("vmf_default", None),
}
_SHIFT = None
_WS = 1.0
_orig_post = GraphVAE._posterior


def _post(self, raw):
    if _SHIFT is not None and self.config.family == "tnbbeta":
        dim = self.config.latent_dim
        s = torch.zeros(raw.shape[-1], dtype=raw.dtype, device=raw.device)
        offsets = (_SHIFT["p"], _SHIFT["q"], _SHIFT["eps"])
        for k, v in zip((dim, dim + 1, dim + 2), offsets, strict=True):
            if v is not None:
                s[k] = v
        raw = raw + s
    return _orig_post(self, raw)


GraphVAE._posterior = _post
_orig_init = GraphVAE.__init__


def _init(self, config):
    _orig_init(self, config)
    if _WS != 1.0 and config.family == "tnbbeta":
        with torch.no_grad():
            self.second.weight.mul_(_WS)


GraphVAE.__init__ = _init

out, fams, lrs, draws, conds = (
    sys.argv[1],
    sys.argv[2].split(","),
    sys.argv[3].split(","),
    [int(x) for x in sys.argv[4].split(",")],
    sys.argv[5].split(","),
)
shard, nshards = (int(sys.argv[6]), int(sys.argv[7])) if len(sys.argv) > 7 else (0, 1)
d = gs.stage_dir("pilot")
cfgs = {c["config_id"]: c for c in json.loads((d / "configs.json").read_text())}
tuned = json.loads((d / "tuned_lr.json").read_text())
dev = select_device(None)
done = set()
if os.path.exists(out):
    for line in Path(out).read_text().splitlines():
        r = json.loads(line)
        done.add((r["family"], r["draw"], r["lr"], r["cond"]))
cells = []
for fam in fams:
    for draw in draws:
        for lrname in lrs:
            lr = tuned[fam]["tnbbeta"] if lrname == "tuned" else float(lrname)
            if lrname == "tuned" and "0.003" in lrs and lr == 0.003:
                continue
            for cname in conds:
                cells.append((fam, draw, lr, cname))
cells = [c for i, c in enumerate(cells) if i % nshards == shard and c not in done]
cache = {}
for fam, draw, lr, cname in cells:
    cid = f"{fam}-00"
    c = cfgs[cid]
    spec = GraphSpec(**{k: v for k, v in c.items() if k != "config_id"})
    seed = {"bipartite": 9002000, "ring": 9014000, "assortative": 9000000}[fam] + draw
    if (fam, draw) not in cache:
        sample = sample_graph(spec, seed)
        cache[(fam, draw)] = (sample, split_edges(sample.adjacency, seed=seed))
    sample, split = cache[(fam, draw)]
    if True:
        if True:
            if True:
                base, sh = CONDS[cname]
                _SHIFT = sh if sh else None
                _WS = sh["ws"] if sh else 1.0
                mcfg = gs.model_config_for(base, int(sample.features.get_shape()[1]))
                traj, state = [], {}
                t0 = time.time()
                tr = run_once(
                    Graph(sample.adjacency, sample.features),
                    split,
                    mcfg,
                    lr=lr,
                    epochs=gs.EPOCHS,
                    seed=seed,
                    device=dev,
                    best_state=state,
                    trajectory=traj,
                )
                model = GraphVAE(mcfg).to(dev)
                model.load_state_dict(state)
                m = gs.evaluate(model, sample, split, seed=seed)
                rb = [(t["epoch"], t["r_bar"]) for t in traj]
                esc = next((e for e, r in rb if r > 0.3), None)
                row = dict(
                    family=fam,
                    draw=draw,
                    seed=seed,
                    lr=lr,
                    cond=cname,
                    targets=(sh or {}).get("targets"),
                    ws=_WS,
                    r_bar_init=rb[0][1],
                    escape_epoch=esc,
                    r_bar_final=traj[-1]["r_bar"],
                    best_epoch=tr["best_epoch"],
                    val_auc=tr["val_auc"],
                    centre_auc=m["centre_auc"],
                    sampled_auc=m["sampled_auc"],
                    ll=m["sampled_log_loss"],
                    recal_ll=m["recal_log_loss"],
                    truth_kl=m["truth_kl"],
                    recal_truth_kl=m["recal_truth_kl"],
                    diverged=tr.get("diverged"),
                    post={k: v for k, v in m.items() if k.startswith("post_")},
                    traj=[
                        {
                            k: t.get(k)
                            for k in (
                                "epoch",
                                "r_bar",
                                "val_auc",
                                "p_mean",
                                "q_mean",
                                "epsilon_mean",
                                "kappa_mean",
                            )
                        }
                        for t in traj
                        if t["epoch"]
                        in (-1, 0, 5, 25, 50, 100, 150, 200, 300, 400, 499)
                        or t["epoch"] % 25 == 0
                    ],
                    seconds=time.time() - t0,
                )
                with open(out, "a") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(
                    fam,
                    draw,
                    lr,
                    cname,
                    f"rbar0 {row['r_bar_init']:.2f} esc {esc} "
                    f"rbarF {row['r_bar_final']:.2f} ep {row['best_epoch']} "
                    f"cAUC {row['centre_auc']:.3f} ll {row['ll']:.3f} "
                    f"recal {row['recal_ll']:.3f}  {row['seconds']:.0f}s",
                    flush=True,
                )
