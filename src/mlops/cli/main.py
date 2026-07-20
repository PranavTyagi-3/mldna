import typer

from mlops import __version__
from mlops.azure import auth as aml_auth
from mlops.azure import blob as aml_blob
from mlops.core import blobauth
from mlops.core.commitcheck import CommitReachabilityError
from mlops.core.conversions import ConversionError
from mlops.core.lockfile import LockfileError
from mlops.core.manifest import ManifestError
from mlops.core.pull import PullError
from mlops.core.push import PushError, push as core_push
from mlops.core.resolve import ResolveError
from mlops.core.rungates import RunGateError
from mlops.core.scaffold import (
    configure_dvc_remote,
    configure_local_dvc_remote,
    has_azure_config,
    has_dvc_config,
    scaffold_consumer,
    scaffold_producer,
    write_azure_config,
    write_azure_tracking_uri,
)
from mlops.core.structure import StructureError
from mlops.doctor import run_doctor
from mlops.fetch import FetchError, fetch as core_fetch
from mlops.publish import PublishError, publish as core_publish
from mlops.pull import pull as core_pull
from mlops.runs import RunsError, list_runs
from mlops.trackinguri import TrackingUriError

PUBLISH_ERRORS = (
    PublishError, ResolveError, RunGateError, CommitReachabilityError,
    StructureError, ConversionError, TrackingUriError,
)
FETCH_ERRORS = (FetchError, LockfileError, ManifestError, TrackingUriError)

app = typer.Typer(
    name="mlops",
    help="Internal MLOps framework combining DVC + MLflow on an Azure ML backend.",
)

configure_app = typer.Typer(help="Configure per-machine auth.")
app.add_typer(configure_app, name="configure")


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


def _choose(prompt_text, labels):
    """Numbered-menu prompt. Returns the 0-based index of the chosen label."""
    typer.echo(prompt_text)
    for i, label in enumerate(labels, 1):
        typer.echo(f"  {i}. {label}")
    while True:
        choice = typer.prompt("Enter a number", type=int)
        if 1 <= choice <= len(labels):
            return choice - 1
        typer.echo(f"Enter a number between 1 and {len(labels)}.")


def _pick_container(list_containers):
    """Lists real containers and lets the developer pick by number, instead of
    typing a name that can be misspelled and silently produce a dead DVC
    remote. Discovery is a convenience, not a hard requirement (unlike AML
    workspace discovery) - RBAC scoped to a single container can legitimately
    fail account-wide listing - so any failure falls back to a manual prompt
    rather than aborting `init`."""
    try:
        containers = list_containers()
    except Exception as e:
        typer.echo(f"Could not list containers: {e}", err=True)
        return typer.prompt("DVC container name")
    if not containers:
        typer.echo("No containers found in that account.")
        return typer.prompt("DVC container name (created on first push)")
    options = containers + ["Enter a new container name"]
    index = _choose("Select DVC container:", options)
    if index == len(containers):
        return typer.prompt("New container name")
    return containers[index]


def _configure_azure_tracking(
    dir, auth_mode, tracking_uri, tenant_id, client_id, client_secret,
    subscription_id, resource_group, workspace,
):
    """Writes mlops.yaml's azure: section and (for service-principal) the local
    auth file. Callers decide *whether* to call this - this function only
    decides *how*, prompting for whatever wasn't already given."""
    # Fully-specified via flags: skip straight to writing, matching today's
    # non-interactive behavior exactly - no auth-mode prompt needed either.
    if subscription_id and resource_group and workspace:
        write_azure_config(dir, subscription_id, resource_group, workspace)
        typer.echo("Wrote azure: section to mlops.yaml")
        return
    if tracking_uri:
        write_azure_tracking_uri(dir, tracking_uri)
        typer.echo("Wrote azure: section to mlops.yaml (tracking_uri)")
        if tenant_id and client_id and client_secret:
            aml_auth.write_local_auth(dir, tenant_id, client_id, client_secret)
            typer.echo(f"Saved to .mlops/{aml_auth.AUTH_FILE}")
        return

    if not auth_mode:
        auth_mode = "service-principal" if (tenant_id or client_id or client_secret) else (
            ["aad", "service-principal"][
                _choose("Select Azure ML authentication mode:", ["Azure AD", "Service Principal"])
            ]
        )

    if auth_mode == "service-principal":
        tenant_id = tenant_id or typer.prompt("Tenant ID")
        client_id = client_id or typer.prompt("Client ID")
        client_secret = client_secret or typer.prompt("Client secret", hide_input=True)
        tracking_uri = typer.prompt("Azure ML MLflow tracking endpoint (Studio -> Consume tab)")
        write_azure_tracking_uri(dir, tracking_uri)
        typer.echo("Wrote azure: section to mlops.yaml (tracking_uri)")
        aml_auth.write_local_auth(dir, tenant_id, client_id, client_secret)
        typer.echo(f"Saved to .mlops/{aml_auth.AUTH_FILE}")
        return

    # aad - discover the workspace unless it was already given explicitly
    subscription_id = subscription_id or typer.prompt("Azure subscription ID")
    if workspace:
        resource_group = resource_group or typer.prompt("Azure resource group")
    else:
        typer.echo("Looking up AML workspaces you have access to...")
        try:
            from azure.identity import DefaultAzureCredential
            from mlops.azure.tracking import list_workspaces
        except ImportError:
            typer.echo(
                "Azure ML workspace discovery needs the `azure` extra. "
                'Fix: pip install "mlops[azure]"',
                err=True,
            )
            raise typer.Exit(1)
        try:
            found = list_workspaces(subscription_id, DefaultAzureCredential())
        except Exception as e:
            typer.echo(f"Could not list workspaces: {e}", err=True)
            raise typer.Exit(1)
        if not found:
            typer.echo("No workspaces found/accessible in that subscription.", err=True)
            raise typer.Exit(1)
        index = _choose("Select AML workspace:", [name for name, _ in found])
        workspace, resource_group = found[index]

    write_azure_config(dir, subscription_id, resource_group, workspace)
    typer.echo("Wrote azure: section to mlops.yaml")


def _configure_dvc(
    dir, model_name, dvc_backend, dvc_local_path,
    dvc_account, dvc_container, dvc_prefix,
    blob_auth_mode, connection_string,
):
    """Returns True if an Azure Blob backend was configured (so the caller
    knows whether to print the shared-secret setup tip)."""
    # Any DVC-related flag given at all signals "scripted call" - infer the
    # backend from what was passed instead of showing a menu, and (below)
    # skip the blob-auth-mode menu too, matching the flag surface that
    # existed before this redesign.
    flags_given = bool(
        dvc_backend or dvc_local_path or dvc_account or dvc_container or dvc_prefix
    )
    if not dvc_backend:
        if dvc_local_path:
            dvc_backend = "local"
        elif dvc_account or dvc_container or dvc_prefix:
            dvc_backend = "azure"
        else:
            dvc_backend = ["local", "azure", "skip"][
                _choose("Select DVC storage backend:", ["Local folder", "Azure Blob", "Skip for now"])
            ]

    if dvc_backend == "skip":
        typer.echo(
            "Skipped DVC remote setup - run `dvc remote add -d <name> <url>` "
            "manually when you're ready; `mlops push` will explain how until then."
        )
        return False

    if dvc_backend == "local":
        default_path = f"../{model_name}-dvc-storage"
        dvc_local_path = dvc_local_path or typer.prompt(
            "Local folder path for DVC storage", default=default_path
        )
        configure_local_dvc_remote(dir, dvc_local_path)
        typer.echo(f"Configured local DVC remote at {dvc_local_path}")
        return False

    if not blob_auth_mode:
        if connection_string:
            blob_auth_mode = "connection-string"
        elif flags_given:
            blob_auth_mode = "aad"
        else:
            blob_auth_mode = ["aad", "connection-string"][
                _choose("Select blob authentication mode:", ["Azure AD", "Connection string"])
            ]

    if blob_auth_mode == "connection-string":
        connection_string = connection_string or typer.prompt("Connection string", hide_input=True)
        if not dvc_account:
            dvc_account = aml_blob.parse_account_name(connection_string)
            if not dvc_account:
                typer.echo(
                    "Could not find AccountName= in that connection string - "
                    "paste the full connection string from Azure Portal > "
                    "Storage account > Access keys.",
                    err=True,
                )
                raise typer.Exit(1)
        dvc_container = dvc_container or _pick_container(
            lambda: aml_blob.list_containers_via_connection_string(connection_string)
        )
    else:
        dvc_account = dvc_account or typer.prompt("DVC storage account name")
        from azure.identity import DefaultAzureCredential
        dvc_container = dvc_container or _pick_container(
            lambda: aml_blob.list_containers_via_credential(dvc_account, DefaultAzureCredential())
        )

    dvc_prefix = dvc_prefix or typer.prompt(
        "Folder/prefix (a container may be shared across models)", default=model_name
    )
    configure_dvc_remote(dir, dvc_account, dvc_container, dvc_prefix)
    typer.echo("Wrote dvc: section to mlops.yaml and configured the DVC remote")

    if blob_auth_mode == "connection-string":
        remote_name = blobauth.get_default_remote(dir)
        blobauth.set_connection_string(dir, remote_name, connection_string)
        typer.echo(f"Saved connection string for remote '{remote_name}' to .dvc/config.local")
    else:
        typer.echo(
            "Using Azure AD (DefaultAzureCredential) for blob access - "
            "run `az login` on this machine if needed."
        )
    return True


@app.command()
def init(
    model_name: str = typer.Argument(None, help="Model name (producer repos only; omit with --consumer)"),
    dir: str = typer.Option(".", "--dir", help="Target project directory"),
    consumer: bool = typer.Option(False, "--consumer", help="Scaffold a consumer repo instead of a producer project"),
    azure: bool = typer.Option(False, "--azure", help="Shortcut: skip the tracking-source menu and configure Azure ML"),
    tracking_source: str = typer.Option(None, "--tracking-source", help="local | azure"),
    auth_mode: str = typer.Option(None, "--auth-mode", help="aad | service-principal"),
    tracking_uri: str = typer.Option(None, "--tracking-uri", help="Azure ML MLflow tracking endpoint (service-principal flow)"),
    tenant_id: str = typer.Option(None, "--tenant-id"),
    client_id: str = typer.Option(None, "--client-id"),
    client_secret: str = typer.Option(None, "--client-secret"),
    subscription_id: str = typer.Option(None, "--subscription-id"),
    resource_group: str = typer.Option(None, "--resource-group"),
    workspace: str = typer.Option(None, "--workspace"),
    dvc_backend: str = typer.Option(None, "--dvc-backend", help="local | azure | skip"),
    dvc_local_path: str = typer.Option(None, "--dvc-local-path"),
    dvc_account: str = typer.Option(None, "--dvc-account"),
    dvc_container: str = typer.Option(None, "--dvc-container"),
    dvc_prefix: str = typer.Option(None, "--dvc-prefix"),
    blob_auth_mode: str = typer.Option(None, "--blob-auth-mode", help="aad | connection-string"),
    connection_string: str = typer.Option(None, "--connection-string"),
) -> None:
    """Scaffold a producer project, or a consumer repo with --consumer."""
    azure_flag_given = bool(
        azure or subscription_id or resource_group or workspace or tracking_uri
        or auth_mode or tenant_id or client_id or client_secret
    )

    if consumer:
        created = scaffold_consumer(dir)
        for path in created:
            typer.echo(f"created {path}")
        typer.echo("Initialized consumer project.")

        if azure_flag_given and not has_azure_config(dir):
            _configure_azure_tracking(
                dir, auth_mode, tracking_uri, tenant_id, client_id, client_secret,
                subscription_id, resource_group, workspace,
            )
            typer.echo(
                "If this machine doesn't have az login access, run "
                "`mlops configure aml-auth` to set a shared secret."
            )
        elif azure_flag_given:
            typer.echo("mlops.yaml already has an azure: section - not re-prompting.")
        return

    if model_name is None:
        typer.echo("model_name is required unless --consumer is set.", err=True)
        raise typer.Exit(1)

    created = scaffold_producer(dir, model_name)
    for path in created:
        typer.echo(f"created {path}")
    typer.echo(f"Initialized producer project '{model_name}'.")

    configured_azure = False
    if has_azure_config(dir):
        typer.echo("mlops.yaml already has an azure: section - not re-prompting.")
    else:
        if not azure_flag_given:
            tracking_source = tracking_source or ["local", "azure"][
                _choose("Select MLflow tracking source:", ["Local", "Azure ML"])
            ]
        else:
            tracking_source = "azure"
        if tracking_source == "azure":
            _configure_azure_tracking(
                dir, auth_mode, tracking_uri, tenant_id, client_id, client_secret,
                subscription_id, resource_group, workspace,
            )
            configured_azure = True

    configured_dvc_azure = False
    if has_dvc_config(dir):
        typer.echo("mlops.yaml already has a dvc: section - not re-prompting.")
    else:
        configured_dvc_azure = _configure_dvc(
            dir, model_name, dvc_backend, dvc_local_path,
            dvc_account, dvc_container, dvc_prefix,
            blob_auth_mode, connection_string,
        )

    if configured_azure or configured_dvc_azure:
        typer.echo(
            "If this machine doesn't have az login access, run "
            "`mlops configure aml-auth` and/or `mlops configure blob-auth` to set "
            "shared secrets."
        )


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


@app.command()
def pull(
    dir: str = typer.Option(".", "--dir", help="Target project directory"),
    run_id: str = typer.Option(
        None, "--run-id", help="Disambiguate when multiple published runs match this commit"
    ),
) -> None:
    """Producer-side: git pull + dvc pull, then pull this commit's published model bundle if any."""

    def _select(candidates):
        if run_id:
            for c in candidates:
                if c.info.run_id == run_id:
                    return c
            typer.echo(
                f"--run-id {run_id} is not among the runs published from this commit.", err=True
            )
            raise typer.Exit(1)
        typer.echo("Multiple published runs match this commit:")
        for i, c in enumerate(candidates):
            typer.echo(
                f"  [{i}] run {c.info.run_id[:8]}  v{c.data.tags.get('registry_version')}  "
                f"published {c.data.tags.get('published_at', '?')}"
            )
        index = typer.prompt("Which one? (index)", type=int)
        return candidates[index]

    try:
        result = core_pull(dir, select=_select)
    except (PullError, TrackingUriError) as e:
        typer.echo(f"pull failed: {e}", err=True)
        raise typer.Exit(1)

    typer.echo("Pulled code and data.")
    if result["model_pulled"]:
        typer.echo(
            f"Pulled published model bundle from run {result['run_id']} "
            f"(v{result['registry_version']}) into models/."
        )
    else:
        typer.echo("No published model found for this commit.")


@app.command()
def publish(
    pkl_path: str,
    formats: list[str] = typer.Option([], "--formats", help="Converted format directories"),
    dir: str = typer.Option(".", "--dir", help="Target project directory"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Register a model version from a tracked run, resolved by content hash."""
    confirm = (lambda summary: True) if yes else (lambda summary: typer.confirm("Proceed?"))

    try:
        result = core_publish(pkl_path, workspace_dir=dir,
                               format_dirs=formats, confirm=confirm)
    except PUBLISH_ERRORS as e:
        typer.echo(f"publish failed: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Published {result['model_name']} v{result['version']} from run {result['run_id']}.")


@app.command()
def fetch(
    dir: str = typer.Option(".", "--dir", help="Target project directory"),
) -> None:
    """Consumer-side: pull pinned model versions listed in models.lock.yaml."""
    try:
        results = core_fetch(dir)
    except FETCH_ERRORS as e:
        typer.echo(f"fetch failed: {e}", err=True)
        raise typer.Exit(1)

    for r in results:
        status = "up to date, skipped" if r["skipped"] else "fetched"
        typer.echo(f"{r['model_name']} v{r['version']}: {status}")


@app.command()
def runs(
    dir: str = typer.Option(".", "--dir", help="Target project directory"),
    last: int = typer.Option(20, "--last", help="Number of most recent runs to show"),
) -> None:
    """Table of recent runs for this project's experiment."""
    try:
        rows = list_runs(dir, last=last)
    except (RunsError, TrackingUriError) as e:
        typer.echo(f"runs failed: {e}", err=True)
        raise typer.Exit(1)

    if not rows:
        typer.echo("No runs found.")
        return

    typer.echo(f"{'RUN ID':<10}{'STARTED':<18}{'COMMIT':<10}{'KEY METRIC':<14}{'DIRTY':<8}SAVE_CONTEXT")
    for r in rows:
        typer.echo(
            f"{r['run_id']:<10}{r['started']:<18}{r['git_commit']:<10}"
            f"{str(r['key_metric']):<14}{r['dirty']:<8}{r['save_context']}"
        )


@configure_app.command("aml-auth")
def configure_aml_auth(
    tenant_id: str = typer.Option(None, "--tenant-id"),
    client_id: str = typer.Option(None, "--client-id"),
    client_secret: str = typer.Option(None, "--client-secret"),
    dir: str = typer.Option(".", "--dir", help="Target project directory"),
    clear: bool = typer.Option(False, "--clear"),
) -> None:
    """Store (or clear) shared service-principal credentials for the Azure ML plane."""
    if clear:
        aml_auth.clear_local_auth(dir)
        typer.echo("Cleared local AML auth.")
    else:
        if tenant_id is None:
            tenant_id = typer.prompt("Tenant ID")
        if client_id is None:
            client_id = typer.prompt("Client ID")
        if client_secret is None:
            client_secret = typer.prompt("Client secret", hide_input=True)
        aml_auth.write_local_auth(dir, tenant_id, client_id, client_secret)
        typer.echo(f"Saved to .mlops/{aml_auth.AUTH_FILE}")

    _, mode = aml_auth.resolve_aml_credential(dir)
    typer.echo(f"Active AML auth mode: {mode}")


@configure_app.command("blob-auth")
def configure_blob_auth(
    connection_string: str = typer.Option(None, "--connection-string"),
    remote: str = typer.Option(None, "--remote", help="DVC remote name (defaults to the project's default remote)"),
    dir: str = typer.Option(".", "--dir", help="Target project directory"),
    clear: bool = typer.Option(False, "--clear"),
) -> None:
    """Store (or clear) a shared connection string for the DVC blob remote."""
    remote_name = remote or blobauth.get_default_remote(dir)
    if not remote_name:
        typer.echo("No DVC remote configured. Run `dvc remote add -d <name> <url>` first.", err=True)
        raise typer.Exit(1)

    if clear:
        blobauth.clear(dir, remote_name)
        typer.echo(f"Cleared local connection string for remote '{remote_name}'.")
    else:
        if connection_string is None:
            connection_string = typer.prompt("Connection string", hide_input=True)
        blobauth.set_connection_string(dir, remote_name, connection_string)
        typer.echo(f"Saved connection string for remote '{remote_name}' to .dvc/config.local")


@app.command()
def doctor(
    dir: str = typer.Option(".", "--dir", help="Target project directory"),
) -> None:
    """Check tool availability, auth, git remote, gitignore, and committed secrets."""
    results = run_doctor(dir)
    failed = False
    for result in results:
        mark = "OK" if result.passed else "FAIL"
        typer.echo(f"[{mark}] {result.name}: {result.message}")
        if not result.passed:
            failed = True

    if failed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
