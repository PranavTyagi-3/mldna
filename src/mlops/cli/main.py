import typer

from mlops import __version__
from mlops.core.push import PushError, push as core_push
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


@app.command()
def push(
    message: str = typer.Option(None, "--message", "-m", help="Commit message"),
    dir: str = typer.Option(".", "--dir", help="Target project directory"),
) -> None:
    """Sync code and data: dvc add, git commit, dvc push, git push."""
    if message is None:
        message = typer.prompt("Commit message")

    try:
        result = core_push(dir, message=message)
    except PushError as e:
        typer.echo(f"push failed: {e}", err=True)
        raise typer.Exit(1)

    if result["committed"]:
        typer.echo(f"Committed: {message}")
    else:
        typer.echo("Nothing to commit.")
    typer.echo("Pushed.")


if __name__ == "__main__":
    app()
