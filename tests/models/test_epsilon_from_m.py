import pytest

from tnbbeta_vae.models.heads import epsilon_from_m


def test_m_zero_is_the_modality_threshold() -> None:
    assert epsilon_from_m(0.0, 16) == 7.5


def test_negative_m_is_below_threshold() -> None:
    assert epsilon_from_m(-6.0, 16) == 1.5


def test_nonpositive_epsilon_raises() -> None:
    with pytest.raises(ValueError, match="must exceed"):
        epsilon_from_m(-7.5, 16)
