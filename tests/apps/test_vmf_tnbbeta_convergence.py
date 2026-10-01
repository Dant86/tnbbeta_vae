"""Tests for apps.distributions.vmf_tnbbeta_convergence."""

from __future__ import annotations

import torch

from apps.distributions.vmf_tnbbeta_convergence import (
    match_tnbbeta_to_vmf,
    run_part1,
    run_part2,
    vmf_mean_resultant_length,
)


class TestVmfMeanResultantLength:
    """Tests for vmf_mean_resultant_length."""

    def test_returns_float(self) -> None:
        """Result is a float."""
        r = vmf_mean_resultant_length(kappa=10.0, dim=3)
        assert isinstance(r, float)

    def test_r_bar_in_valid_range(self) -> None:
        """r_bar is in (0, 1] for any kappa > 0, dim >= 2."""
        for kappa in [1.0, 10.0, 100.0]:
            for dim in [3, 5, 11]:
                r = vmf_mean_resultant_length(kappa, dim)
                assert 0 < r <= 1

    def test_r_bar_increases_with_kappa(self) -> None:
        """r_bar increases monotonically with concentration kappa."""
        dim = 5
        kappas = [1.0, 10.0, 50.0, 100.0]
        r_bars = [vmf_mean_resultant_length(k, dim) for k in kappas]

        for i in range(len(r_bars) - 1):
            assert r_bars[i] < r_bars[i + 1]

    def test_r_bar_decreases_with_dimension(self) -> None:
        """r_bar decreases as dimension increases (for fixed kappa)."""
        kappa = 50.0
        dims = [3, 5, 7, 11]
        r_bars = [vmf_mean_resultant_length(kappa, d) for d in dims]

        for i in range(len(r_bars) - 1):
            assert r_bars[i] > r_bars[i + 1]

    def test_r_bar_matches_known_value(self) -> None:
        """Spot-check against a known value (vMF on S^2, kappa=50)."""
        # For S^2 (dim=3), kappa=50, r_bar should be roughly 0.98
        # This is a rough sanity check, not a precise value
        r = vmf_mean_resultant_length(kappa=50.0, dim=3)
        assert 0.97 < r <= 1.0


class TestMatchTnbbetaToVmf:
    """Tests for match_tnbbeta_to_vmf."""

    def test_returns_float_in_valid_range(self) -> None:
        """Returned p is a float in (p_min, p_max)."""
        target_r_bar = 0.8
        p = match_tnbbeta_to_vmf(
            target_r_bar,
            epsilon=1.0,
            dim=3,
            q=0.0,
            sample_size=10000,
            bisect_tol=1e-2,
        )
        assert isinstance(p, float)
        assert 0.5 < p < 1.0

    def test_matched_r_bar_is_close_to_target(self) -> None:
        """Empirical r_bar from matched TNBBeta is close to target."""
        target_r_bar = 0.7
        epsilon = 1.0
        dim = 5
        sample_size = 50000
        bisect_tol = 1e-3

        p = match_tnbbeta_to_vmf(
            target_r_bar,
            epsilon=epsilon,
            dim=dim,
            q=0.0,
            sample_size=sample_size,
            bisect_tol=bisect_tol,
        )

        # Verify the match by computing empirical r_bar
        from tnbbeta_vae.distributions.tnbbeta_spherical import TNBBetaSpherical

        mu = torch.zeros(dim, dtype=torch.float64)
        mu[0] = 1.0
        dist = TNBBetaSpherical(mu, p=p, q=0.0, epsilon=epsilon)
        samples = dist.rsample(torch.Size([sample_size]))
        empirical_r_bar = float(torch.norm(samples.mean(dim=0)).item())

        # Allow some tolerance for sampling noise
        assert abs(empirical_r_bar - target_r_bar) < 0.02

    def test_matched_p_increases_with_target_r_bar(self) -> None:
        """Matched p increases as target r_bar increases."""
        epsilon = 1.0
        dim = 3
        sample_size = 10000
        bisect_tol = 1e-2

        p_low = match_tnbbeta_to_vmf(
            0.5,
            epsilon=epsilon,
            dim=dim,
            sample_size=sample_size,
            bisect_tol=bisect_tol,
        )
        p_high = match_tnbbeta_to_vmf(
            0.8,
            epsilon=epsilon,
            dim=dim,
            sample_size=sample_size,
            bisect_tol=bisect_tol,
        )

        assert p_low < p_high

    def test_works_with_different_dimensions(self) -> None:
        """Bisection succeeds for various dimensions."""
        target_r_bar = 0.75
        epsilon = 1.0
        sample_size = 10000
        bisect_tol = 1e-2

        for dim in [3, 5, 11]:
            p = match_tnbbeta_to_vmf(
                target_r_bar,
                epsilon=epsilon,
                dim=dim,
                sample_size=sample_size,
                bisect_tol=bisect_tol,
            )
            assert 0.5 < p < 1.0

    def test_works_with_different_epsilons(self) -> None:
        """Bisection succeeds for various epsilon values."""
        target_r_bar = 0.75
        dim = 5
        sample_size = 10000
        bisect_tol = 1e-2

        for epsilon in [0.5, 1.0, 2.0]:
            p = match_tnbbeta_to_vmf(
                target_r_bar,
                epsilon=epsilon,
                dim=dim,
                sample_size=sample_size,
                bisect_tol=bisect_tol,
            )
            assert 0.5 < p < 1.0

    def test_matches_high_kappa_targets(self) -> None:
        """Regression test: very high vMF kappa (r_bar near 1) is reachable.

        With the old ``p_max=0.9999`` default, ``r_bar`` topped out around
        0.998 regardless of epsilon, so this target (vMF's r_bar at
        kappa=1000, S^2) could not be matched within tolerance.
        """
        dim = 3
        target_r_bar = vmf_mean_resultant_length(kappa=1000.0, dim=dim)

        p = match_tnbbeta_to_vmf(
            target_r_bar,
            epsilon=1.0,
            dim=dim,
            sample_size=100000,
            bisect_tol=1e-4,
        )

        from tnbbeta_vae.distributions.tnbbeta_spherical import TNBBetaSpherical

        mu = torch.zeros(dim, dtype=torch.float64)
        mu[0] = 1.0
        dist = TNBBetaSpherical(mu, p=p, q=0.0, epsilon=1.0)
        samples = dist.rsample((200000,))
        achieved_r_bar = float(torch.norm(samples.mean(dim=0)).item())

        assert abs(achieved_r_bar - target_r_bar) < 5e-3

    def test_adjacent_high_kappa_targets_give_distinct_p(self) -> None:
        """Regression test: close targets resolve to distinct matched p.

        vMF's r_bar at kappa=500 vs. kappa=1000 (S^2) differ by only
        0.001; the default ``bisect_tol`` must be tight enough that these
        don't collapse onto the same matched ``p``.
        """
        dim = 3
        r_500 = vmf_mean_resultant_length(kappa=500.0, dim=dim)
        r_1000 = vmf_mean_resultant_length(kappa=1000.0, dim=dim)

        p_500 = match_tnbbeta_to_vmf(r_500, epsilon=1.0, dim=dim, sample_size=100000)
        p_1000 = match_tnbbeta_to_vmf(r_1000, epsilon=1.0, dim=dim, sample_size=100000)

        assert p_500 < p_1000

    def test_deterministic_given_fixed_seed(self) -> None:
        """Same inputs (including seed) reproduce the same matched p.

        The common-random-numbers trick (reseeding before every candidate
        draw) is what makes the bisection a true monotonic root-find
        rather than one perturbed by independent sampling noise at each
        candidate; this also makes the whole call reproducible.
        """

        def _match() -> float:
            return match_tnbbeta_to_vmf(
                0.9,
                epsilon=1.0,
                dim=5,
                sample_size=10000,
                bisect_tol=1e-3,
                seed=42,
            )

        assert _match() == _match()


class TestConvergenceBehavior:
    """Integration tests for convergence properties."""

    def test_gap_shrinks_as_kappa_increases(self) -> None:
        """Two-sample gap decreases as vMF concentration increases."""
        from apps.distributions.vmf_tnbbeta_convergence import (
            _compute_two_sample_stats,
            _w_marginal_from_samples,
        )
        from tnbbeta_vae.distributions.von_mises_fisher import VonMisesFisher

        epsilon = 1.0
        dim = 3
        sample_size = 20000
        bisect_tol = 1e-2

        device = torch.device("cpu")
        mu = torch.zeros(dim, dtype=torch.float64, device=device)
        mu[0] = 1.0
        mu_np = mu.numpy()

        gaps = []
        for kappa in [5.0, 20.0, 100.0]:
            target_r_bar = vmf_mean_resultant_length(kappa, dim)

            p = match_tnbbeta_to_vmf(
                target_r_bar,
                epsilon=epsilon,
                dim=dim,
                sample_size=sample_size,
                bisect_tol=bisect_tol,
            )

            vmf_dist = VonMisesFisher(
                loc=mu.unsqueeze(0),
                scale=torch.tensor([[kappa]], dtype=torch.float64),
            )

            from tnbbeta_vae.distributions.tnbbeta_spherical import TNBBetaSpherical

            tnbbeta_dist = TNBBetaSpherical(mu, p=p, q=0.0, epsilon=epsilon)

            vmf_samples = vmf_dist.rsample(torch.Size([sample_size])).squeeze(1).numpy()
            tnbbeta_samples = tnbbeta_dist.rsample(torch.Size([sample_size])).numpy()

            w_vmf = _w_marginal_from_samples(vmf_samples, mu_np)
            w_tnbbeta = _w_marginal_from_samples(tnbbeta_samples, mu_np)

            gap_energy, _ = _compute_two_sample_stats(w_vmf, w_tnbbeta)
            gaps.append(gap_energy)

        # Gaps should generally decrease (allowing some noise)
        assert gaps[0] > gaps[-1] or abs(gaps[0] - gaps[-1]) < 0.01


class TestRunPart1:
    """Smoke tests for the Part 1 driver."""

    def test_returns_one_result_per_kappa_epsilon_pair(self) -> None:
        """run_part1 returns a ConvergenceResult for every grid point."""
        dimensions = [3]
        kappas = [10.0, 100.0]
        epsilons = [1.0, 2.0]

        results = run_part1(
            dimensions=dimensions,
            kappas=kappas,
            epsilons=epsilons,
            sample_size=5000,
            bisect_tol=1e-2,
        )

        assert len(results) == len(dimensions) * len(kappas) * len(epsilons)
        for result in results:
            assert result.gap_energy >= 0
            assert 0.0 <= result.gap_ks <= 1.0

    def test_gap_energy_exceeds_noise_floor(self) -> None:
        """The TNBBeta/vMF gap should be larger than the vMF/vMF noise floor."""
        results = run_part1(
            dimensions=[3],
            kappas=[20.0],
            epsilons=[1.0],
            sample_size=20000,
            bisect_tol=1e-3,
        )

        assert len(results) == 1
        assert results[0].gap_energy > results[0].noise_floor_energy


class TestRunPart2:
    """Tests for the Part 2 checkpoint-availability check."""

    def test_reports_missing_checkpoint_dir(self, tmp_path, monkeypatch) -> None:
        """Returns a clear status, not fabricated numbers, when unavailable."""
        missing_dir = tmp_path / "does_not_exist"
        monkeypatch.setattr(
            "apps.distributions.vmf_tnbbeta_convergence.checkpoint_dir",
            lambda: missing_dir,
        )

        result = run_part2()

        assert result["status"] == "checkpoint_dir_not_found"

    def test_reports_no_matching_checkpoints(self, tmp_path, monkeypatch) -> None:
        """Returns a clear status when the checkpoint dir has no sweep runs."""
        (tmp_path / "unrelated_run").mkdir()
        monkeypatch.setattr(
            "apps.distributions.vmf_tnbbeta_convergence.checkpoint_dir",
            lambda: tmp_path,
        )

        result = run_part2()

        assert result["status"] == "no_checkpoints"
