"""A learned diffusion prior for ConvTNBBetaSphericalVAE's latent space.

Wraps a frozen, pretrained ConvTNBBetaSphericalVAE checkpoint and a small
MLP denoiser trained by direct z_0-regression against the closed-form
forward-noising process in tnbbeta_vae.diffusion. Kept frozen rather than
trained jointly -- see
docs/superpowers/specs/2026-10-05-sphere-diffusion-prior-design.md point 8
for why (citing Rombach et al. 2022's explicit finding against joint
training, and this project's own need for an attributable comparison
against the VAE's existing fixed-prior baseline).
"""

from __future__ import annotations

from pydantic import BaseModel
import torch
from torch import Tensor, nn

from tnbbeta_vae.diffusion.noising import noise_to
from tnbbeta_vae.diffusion.schedule import DiffusionSchedule
from tnbbeta_vae.distributions import TNBBetaSpherical
from tnbbeta_vae.models.architectures.denoiser_mlp import SphereDenoiserMLP
from tnbbeta_vae.models.conv_vae import ConvTNBBetaSphericalVAE
from tnbbeta_vae.models.priors.tnbbeta_spherical import uniform_prior_params
from tnbbeta_vae.paths import checkpoint_dir
from tnbbeta_vae.registry import register_model

__all__ = ["SphereDiffusionPrior", "SphereDiffusionPriorConfig"]


class SphereDiffusionPriorConfig(BaseModel):
    """Hyperparameters for :class:`SphereDiffusionPrior`.

    Attributes:
        vae_run_name: Checkpoint directory name (under
            ``$TNBBETA_CHECKPOINT_DIR``) of the frozen, pretrained
            ConvTNBBetaSphericalVAE to build this prior on top of.
        q_target: Forward process's target concentration; can approach but
            never reach 0 (NB(r,q) requires q < 1).
        p_start: Near-point-mass starting median, close to 1.
        eps_start: Near-point-mass starting size parameter, deliberately
            large -- a fixed diffusion hyperparameter, not read from the
            VAE's own per-example posterior uncertainty.
        t_min: Smallest diffusion time training samples t from; kept above
            0 so the denoising task near t=0 isn't trivial (predicting z_0
            from z_t ~= z_0 carries no training signal).
        t_max: Largest diffusion time training samples t from, and the
            reverse sampler's starting time.
        speed: Leisen et al.'s rate constant.
        num_reverse_steps: Number of predict-then-renoise steps in generate().
        denoiser_hidden_dim: Width of the denoiser MLP's hidden layers.
        denoiser_depth: Number of the denoiser MLP's hidden layers.
    """

    vae_run_name: str
    q_target: float = 0.05
    p_start: float = 0.999
    eps_start: float = 50.0
    t_min: float = 0.01
    t_max: float = 10.0
    speed: float = 1.0
    num_reverse_steps: int = 50
    denoiser_hidden_dim: int = 256
    denoiser_depth: int = 4


@register_model(
    "tnbbeta_spherical_diffusion_prior", config_cls=SphereDiffusionPriorConfig
)
class SphereDiffusionPrior(nn.Module):
    """A learned prior over ConvTNBBetaSphericalVAE's latent sphere."""

    def __init__(self, config: SphereDiffusionPriorConfig) -> None:
        """Initializes the model, loading and freezing the pretrained VAE.

        Args:
            config: Hyperparameters; see :class:`SphereDiffusionPriorConfig`.

        Raises:
            FileNotFoundError: If no checkpoint exists at ``config.vae_run_name``
                (e.g. a preempted run that only ever wrote ``latest.pt``).
            TypeError: If the checkpoint at ``config.vae_run_name`` is not a
                ``ConvTNBBetaSphericalVAE``.
        """
        super().__init__()
        self.config = config

        # Imported here, not at module level: tnbbeta_vae.training.checkpoint
        # imports tnbbeta_vae.models for its @register_model side effects, and
        # tnbbeta_vae.models imports this module -- a module-level import here
        # would make `import tnbbeta_vae.training` fail with a circular import.
        from tnbbeta_vae.training.checkpoint import load_model_checkpoint

        vae_path = checkpoint_dir() / config.vae_run_name / "final.pt"
        try:
            vae, _ = load_model_checkpoint(vae_path)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"SphereDiffusionPrior could not find a VAE checkpoint for "
                f"vae_run_name={config.vae_run_name!r} at {vae_path} -- check that "
                "this run name is correct and that it finished with a 'final.pt' "
                "(a preempted run that only ever wrote 'latest.pt' cannot host a "
                "diffusion prior yet). Checkpoints are looked up under "
                "$TNBBETA_CHECKPOINT_DIR (see tnbbeta_vae.paths.checkpoint_dir), "
                f"currently resolving to {checkpoint_dir()}."
            ) from error
        if not isinstance(vae, ConvTNBBetaSphericalVAE):
            raise TypeError(
                "SphereDiffusionPrior requires a ConvTNBBetaSphericalVAE checkpoint "
                f"at vae_run_name={config.vae_run_name!r}, got {type(vae).__name__}."
            )
        self.vae = vae
        for param in self.vae.parameters():
            param.requires_grad_(False)

        self.latent_dim = vae.config.latent_dim
        eps_target = uniform_prior_params(self.latent_dim)[2]
        self.schedule = DiffusionSchedule(
            eps_target=eps_target, q_target=config.q_target, speed=config.speed
        )
        self.denoiser = SphereDenoiserMLP(
            self.latent_dim, config.denoiser_hidden_dim, config.denoiser_depth
        )

    def train(self, mode: bool = True) -> SphereDiffusionPrior:
        """Puts the denoiser in train/eval mode; the frozen VAE always stays eval."""
        self.training = mode
        self.denoiser.train(mode)
        self.vae.eval()
        return self

    def forward(self, x: Tensor) -> tuple[Tensor, TNBBetaSpherical, Tensor]:
        """Delegates to the frozen VAE's own ``forward`` (encode/sample/decode).

        This diffusion prior only replaces the VAE's *prior* at generation
        time (see :meth:`generate`); reconstruction from a real input goes
        through the VAE exactly as it always did.
        """
        return self.vae.forward(x)

    def posterior_and_prior(
        self, x: Tensor
    ) -> tuple[TNBBetaSpherical, TNBBetaSpherical]:
        """Delegates to the frozen VAE's own ``posterior_and_prior``.

        The VAE's posterior is genuinely what eval scripts probing this
        model's latent space want; this diffusion prior only replaces the
        VAE's fixed uniform *prior* at generation time, not its posterior.
        """
        return self.vae.posterior_and_prior(x)

    def log_likelihood(self, x: Tensor, z: Tensor) -> Tensor:
        """Delegates to the frozen VAE's own ``log_likelihood``."""
        return self.vae.log_likelihood(x, z)

    def training_step(self, batch: Tensor, kl_weight: float = 1.0) -> dict[str, Tensor]:
        """Computes the z_0-regression loss for one batch of images.

        Args:
            batch: Input images, shape ``(batch_size, 3, image_size, image_size)``.
            kl_weight: Unused. Accepted only so ``Trainer.fit(...,
                kl_warmup_epochs=N)`` doesn't crash when run against this
                model -- there is no KL term in this regression objective
                for a warm-up weight to apply to.

        Returns:
            A dict with ``"loss"`` (mean ``1 - cosine_similarity``).
        """
        with torch.no_grad():
            posterior = self.vae.posterior_and_prior(batch)[0]
            z_0 = posterior.rsample()
        z_0 = z_0.detach()

        batch_size = z_0.shape[0]
        t = (
            torch.distributions.Uniform(self.config.t_min, self.config.t_max)
            .sample((batch_size,))
            .to(z_0.device)
        )
        z_t = noise_to(
            z_0, t, self.schedule, self.config.p_start, 0.0, self.config.eps_start
        )
        z_0_hat = self.denoiser(z_t, t)
        loss = 1 - torch.nn.functional.cosine_similarity(z_0_hat, z_0, dim=-1)
        return {"loss": loss.mean()}

    @torch.no_grad()
    def generate(self, num_samples: int) -> Tensor:
        """Generates images via reverse (predict-then-renoise) diffusion.

        Args:
            num_samples: Number of images to generate.

        Returns:
            Images of shape ``(num_samples, 3, image_size, image_size)``.
        """
        z = self.vae.prior().sample(torch.Size([num_samples]))
        steps = torch.linspace(
            self.config.t_max,
            0.0,
            self.config.num_reverse_steps + 1,
            device=z.device,
        )
        for i in range(self.config.num_reverse_steps):
            t_current, t_next = steps[i], steps[i + 1]
            z_0_hat = self.denoiser(z, t_current.expand(num_samples))
            if t_next <= 0.0:
                z = z_0_hat
            else:
                z = noise_to(
                    z_0_hat, t_next.expand(num_samples), self.schedule,
                    self.config.p_start, 0.0, self.config.eps_start,
                )  # fmt: skip
        return self.vae.decoder(z)
