"""Command-line entry point. Subcommands are added phase by phase."""

import typer

from modeldetective import __version__

app = typer.Typer(no_args_is_help=True, help="Detect LLM quantization from text outputs.")


@app.callback()
def main() -> None:
    """Detect LLM quantization from text outputs."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
