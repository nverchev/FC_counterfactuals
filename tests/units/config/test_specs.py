"""Tests for the config specs module."""

import dataclasses
import pathlib
import pytest

from src.config.environment import VERSION
from src.config.specs import PathSpecs
from src.config import get_experiment_config


class TestPathSpecs:
    """Test PathSpecs version directory computation."""

    def test_version_dir(self) -> None:
        """Test version_dir path construction."""
        specs = PathSpecs(root_exp_dir=pathlib.Path('./exp_test'))
        assert specs.version_dir == pathlib.Path(f'./exp_test/v{VERSION}')
        return


class TestAllConfig:
    """Test ExperimentConfig properties and variation handling."""

    def test_resolved_variation_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolved_variation uses git_branch if variation is None."""
        monkeypatch.setattr('src.config.specs.get_current_branch', lambda: 'main')
        cfg = dataclasses.replace(get_experiment_config(), variation=None)
        assert cfg.resolved_variation == 'main'
        return

    def test_resolved_variation_explicit(self) -> None:
        """Test resolved_variation returns explicit variation string."""
        cfg = dataclasses.replace(get_experiment_config(), variation='custom_var')
        assert cfg.resolved_variation == 'custom_var'
        return

    def test_name_property_with_final_flag(self) -> None:
        """Test experiment name suffix when final flag is set."""
        cfg = dataclasses.replace(get_experiment_config(), variation='test_run', final=True)
        assert cfg.name == 'test_run_final'
        return
