"""Smoke tests: the package imports and the CLI answers. No GPU, no models."""

from typer.testing import CliRunner

from modeldetective import __version__
from modeldetective.cli import app


def test_version_string() -> None:
    assert __version__ == "0.1.0"


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout
