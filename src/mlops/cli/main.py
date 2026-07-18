import typer

from mlops import __version__
from mlops.core.scaffold import scaffold_producer

app = typer.Typer(
    name="mlops",
    help="Internal MLOps framework combining DVC + MLflow on an Azure ML backend.",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True,
        help="Show the mlops version and exit.",
    ),
) -> None:
    pass


@app.command()
def init(
    model_name: str,
    dir: str = typer.Option(".", "--dir", help="Target project directory"),
) -> None:
    """Scaffold a producer project."""
    created = scaffold_producer(dir, model_name)
    for path in created:
        typer.echo(f"created {path}")
    typer.echo(f"Initialized producer project '{model_name}'.")


if __name__ == "__main__":
    app()
