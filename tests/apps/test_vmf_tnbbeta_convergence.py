"""Tests for apps.distributions.vmf_tnbbeta_convergence."""

from __future__ import annotations

import torch

from apps.distributions.vmf_tnbbeta_convergence import (
    match_tnbbeta_to_vmf,
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

            vmf_samples = vmf_dist.rsample(torch.Size([sample_size])).squeeze(0).numpy()
            tnbbeta_samples = tnbbeta_dist.rsample(torch.Size([sample_size])).numpy()

            w_vmf = _w_marginal_from_samples(vmf_samples, mu_np)
            w_tnbbeta = _w_marginal_from_samples(tnbbeta_samples, mu_np)

            gap_energy, _ = _compute_two_sample_stats(w_vmf, w_tnbbeta)
            gaps.append(gap_energy)

        # Gaps should generally decrease (allowing some noise)
        assert gaps[0] > gaps[-1] or abs(gaps[0] - gaps[-1]) < 0.01
