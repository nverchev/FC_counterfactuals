"""Tests for the Hydra integration module."""

import pytest

from src.config.hydra import get_experiment_config
from src.config.experiment import get_variation


class TestHydraHelpers:
    """Test Hydra config composition and override tracking."""

    def test_get_config_all_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test getting default typed configuration."""
        monkeypatch.setattr('src.config.specs.get_current_branch', lambda: 'main')
        cfg = get_experiment_config()
        assert cfg.user.device == 'auto'
        return

    def test_get_config_all_with_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test applying overrides to get_experiment_config."""
        monkeypatch.setattr('src.config.specs.get_current_branch', lambda: 'main')
        overrides = ['classification.train.n_epochs=25']
        cfg = get_experiment_config(overrides=overrides)
        assert cfg.classification.train.n_epochs == 25
        assert cfg.tags == ['classification.train.n_epochs=25']
        return

    def test_get_config_all_starts_from_the_named_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The entry configuration is recorded, and its runs get their own folder."""
        monkeypatch.setattr('src.config.specs.get_current_branch', lambda: 'main')
        cfg = get_experiment_config('mlp')
        assert cfg.user.hydra.config_name == 'mlp'
        assert cfg.par_dir == cfg.user.path.version_dir / 'mlp'
        return

    def test_get_variation_filters_user_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_variation ignores the user overrides."""
        monkeypatch.setattr('src.config.experiment.get_current_branch', lambda: 'main')
        overrides = ['user.device=cpu', 'classification.train.n_epochs=50']
        variation, tags = get_variation(None, overrides)
        assert variation == 'main_classification.train.n_epochs=50'
        assert tags == ['classification.train.n_epochs=50']
        return
