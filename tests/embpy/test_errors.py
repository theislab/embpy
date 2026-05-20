"""Tests for custom exception classes."""

import pytest

from embpy.errors import ConfigError, IdentifierError, ModelNotFoundError


class TestConfigError:
    def test_default_message(self):
        err = ConfigError()
        assert str(err) == "Invalid configuration"

    def test_custom_message(self):
        err = ConfigError("Bad device setting")
        assert str(err) == "Bad device setting"
        assert err.message == "Bad device setting"

    def test_is_exception(self):
        with pytest.raises(ConfigError):
            raise ConfigError("test")


class TestIdentifierError:
    def test_default_message(self):
        err = IdentifierError()
        assert str(err) == "Invalid identifier"

    def test_custom_message(self):
        err = IdentifierError("Gene not found: FAKEGENE")
        assert "FAKEGENE" in str(err)

    def test_is_exception(self):
        with pytest.raises(IdentifierError):
            raise IdentifierError("test")


class TestModelNotFoundError:
    def test_default_message(self):
        err = ModelNotFoundError()
        assert str(err) == "Model not found"

    def test_custom_message(self):
        err = ModelNotFoundError("Model 'fake_model' not in registry")
        assert "fake_model" in str(err)

    def test_is_exception(self):
        with pytest.raises(ModelNotFoundError):
            raise ModelNotFoundError("test")
