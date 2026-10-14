import sys
from pathlib import Path

import typer

from mldna import __version__
from mldna.attach import AttachError, attach as core_attach
from mldna.azure import auth as aml_auth
from mldna.azure import blob as aml_blob
from mldna.core import blobauth
from mldna.core.commitcheck import CommitReachabilityError
from mldna.core.config import ProjectConfig, find_parent_project, missing_project_hint
from mldna.core.conversions import ConversionError
from mldna.core.lockfile import LockfileError
from mldna.core.manifest import ManifestError
from mldna.core.pull import PullError
from mldna.core.push import PushError, check_something_to_push, push as core_push, stage as core_stage
from mldna.core.resolve import ResolveError, describe_candidate
from mldna.core.rungates import RunGateError
from mldna.core.scaffold import (
    configure_dvc_remote,
    configure_local_dvc_remote,
    ensure_model_id,
    has_azure_config,
    has_dvc_config,
    has_filesystem_config,
    has_mlflow_config,
    scaffold_consumer,
    scaffold_producer,
    write_azure_config,
    write_azure_tracking_uri,
    write_filesystem_path,
    write_mlflow_tracking_uri,
)
from mldna.core.structure import StructureError
from mldna.core import workspace as ws
from mldna.doctor import parse_timeout, run_doctor
from mldna.fetch import FetchError, fetch as core_fetch
from mldna.publish import PublishError, publish as core_publish
from mldna.pull import pull as core_pull
from mldna.runs import RunsError, list_runs
from mldna.trackinguri import TrackingUriError

PUBLISH_ERRORS = (
    PublishError, ResolveError, RunGateError, CommitReachabilityError,
    StructureError, ConversionError, TrackingUriError,
)
FETCH_ERRORS = (FetchError, LockfileError, ManifestError, TrackingUriError)
ATTACH_ERRORS = (AttachError, TrackingUriError)

# Order matches the init menu; "local" writes no tracking section at all.
TRACKING_SOURCES = ("local", "mlflow", "azure", "filesystem")

app = typer.Typer(
    name="mldna",
    help="A lineage layer for ML teams: DVC + MLflow glue with content-addressed model lineage, gated publishing, and lockfile-pinned consumption.",
)

configure_app = typer.Typer(help="Configure per-machine auth.")
app.add_typer(configure_app, name="configure")
workspace_app = typer.Typer(help="Parallel working copies (git worktrees).")
app.add_typer(workspace_app, name="workspace")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True,
        help="Show the mldna version and exit.",
    ),
) -> None:
    # Line-buffered so an error (stderr) never lands above its context (stdout) in CI logs.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except AttributeError:
            pass


def _stdin_is_interactive() -> bool:
    """A closed/piped stdin (e.g. CI) can never answer a prompt - detect it
    once here rather than letting each prompting command hang or crash."""
    if not sys.stdin.isatty():
        return False
    if sys.platform != "win32":
        return True
    # Windows reports NUL as a tty; only a real console handle can answer prompts.
    try:
        import ctypes
        import msvcrt
        handle = msvcrt.get_osfhandle(sys.stdin.fileno())
        return bool(ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(ctypes.c_uint32())))
    except Exception:
        return False


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
    """Writes mldna.yaml's azure: section and (for service-principal) the local
    auth file. Callers decide *whether* to call this - this function only
    decides *how*, prompting for whatever wasn't already given."""
    # Fully-specified via flags: skip straight to writing, matching today's
    # non-interactive behavior exactly - no auth-mode prompt needed either.
    if subscription_id and resource_group and workspace:
        write_azure_config(dir, subscription_id, resource_group, workspace)
        typer.echo("Wrote azure: section to mldna.yaml")
        return
    if tracking_uri:
        write_azure_tracking_uri(dir, tracking_uri)
        typer.echo("Wrote azure: section to mldna.yaml (tracking_uri)")
        if tenant_id and client_id and client_secret:
            aml_auth.write_local_auth(dir, tenant_id, client_id, client_secret)
            typer.echo(f"Saved to .mldna/{aml_auth.AUTH_FILE}")
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
        typer.echo("Wrote azure: section to mldna.yaml (tracking_uri)")
        aml_auth.write_local_auth(dir, tenant_id, client_id, client_secret)
        typer.echo(f"Saved to .mldna/{aml_auth.AUTH_FILE}")
        return

    # aad - discover the workspace unless it was already given explicitly
    subscription_id = subscription_id or typer.prompt("Azure subscription ID")
    if workspace:
        resource_group = resource_group or typer.prompt("Azure resource group")
    else:
        typer.echo("Looking up AML workspaces you have access to...")
        try:
            from azure.identity import DefaultAzureCredential
            from mldna.azure.tracking import list_workspaces
        except ImportError:
            typer.echo(
                "Azure ML workspace discovery needs the `azure` extra. "
                'Fix: pip install "mldna-lineage[azure]"',
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
    typer.echo("Wrote azure: section to mldna.yaml")


def _non_azure_remote_message(remote_name, remote_url):
    """A blob connection string is only ever read for an azure:// remote, so
    writing one anywhere else is meaningless config. Same remote-scheme class
    as doctor's check_blob_auth."""
    return (
        f"DVC remote '{remote_name}' is not an Azure Blob remote "
        f"(url: {remote_url or 'none configured'}), so a blob connection string "
        "would never be read - DVC reads connection_string only for azure:// "
        "remotes. A local or other non-Azure remote is a fully supported setup "
        "that needs no blob auth. Fix: nothing to configure for "
        f"'{remote_name}' - or name the Azure remote explicitly: mldna configure "
        "blob-auth --remote <name> (list remotes with `dvc remote list`)."
    )


def _configure_filesystem_tracking(dir, filesystem_path):
    """Writes mldna.yaml's filesystem: section - an existing MLflow file-based
    store on disk, shared between a producer and its consumers (D#39). The
    path is not validated here: trackinguri resolves it (expanding variables
    and repo-relative paths) and refuses on a missing store with the resolved
    absolute path in the message."""
    filesystem_path = filesystem_path or typer.prompt(
        "Path to an existing MLflow file store (absolute, repo-relative, or "
        "with env vars, e.g. ${MLDNA_TRACKING_ROOT}/mlruns)"
    )
    write_filesystem_path(dir, filesystem_path)
    typer.echo("Wrote filesystem: section to mldna.yaml")


def _configure_mlflow_tracking(dir, mlflow_uri):
    """Writes mldna.yaml's mlflow: section - a plain, non-Azure MLflow tracking
    server URL. No mldna-managed auth: mlflow-skinny itself reads
    MLFLOW_TRACKING_USERNAME/PASSWORD/TOKEN natively for servers that need it."""
    mlflow_uri = mlflow_uri or typer.prompt("MLflow tracking server URL (e.g. http://host:5000)")
    write_mlflow_tracking_uri(dir, mlflow_uri)
    typer.echo("Wrote mlflow: section to mldna.yaml")


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
            "manually when you're ready; `mldna push` will explain how until then."
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
        try:
            from azure.identity import DefaultAzureCredential
        except ImportError:
            typer.echo(
                "An Azure Blob DVC remote needs the `azure` extra. "
                'Fix: pip install "mldna-lineage[azure]"',
                err=True,
            )
            raise typer.Exit(1)
        dvc_container = dvc_container or _pick_container(
            lambda: aml_blob.list_containers_via_credential(dvc_account, DefaultAzureCredential())
        )

    dvc_prefix = dvc_prefix or typer.prompt(
        "Folder/prefix (a container may be shared across models)", default=model_name
    )
    configure_dvc_remote(dir, dvc_account, dvc_container, dvc_prefix)
    typer.echo("Wrote dvc: section to mldna.yaml and configured the DVC remote")

    if blob_auth_mode == "connection-string":
        # configure_dvc_remote leaves an existing default remote alone, so the
        # default remote here is not necessarily the azure:// one.
        remote_name = blobauth.get_default_remote(dir)
        remote_url = blobauth.get_remote_url(dir, remote_name)
        if not blobauth.is_azure_remote(remote_url):
            typer.echo(_non_azure_remote_message(remote_name, remote_url), err=True)
        else:
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
    namespace: str = typer.Option(
        None, "--namespace",
        help="Prefix for the experiment/registry name (producer only; default: derived from the project directory name)",
    ),
    consumer: bool = typer.Option(False, "--consumer", help="Scaffold a consumer repo instead of a producer project"),
    azure: bool = typer.Option(False, "--azure", help="Shortcut: skip the tracking-source menu and configure Azure ML"),
    tracking_source: str = typer.Option(
        None, "--tracking-source", help="local | mlflow | azure | filesystem",
    ),
    auth_mode: str = typer.Option(None, "--auth-mode", help="aad | service-principal"),
    tracking_uri: str = typer.Option(None, "--tracking-uri", help="Azure ML MLflow tracking endpoint (service-principal flow)"),
    mlflow_uri: str = typer.Option(None, "--mlflow-uri", help="Plain (non-Azure) MLflow tracking server URL"),
    filesystem_path: str = typer.Option(
        None, "--filesystem-path",
        help="Path to an existing MLflow file-based store shared between repos",
    ),
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
    # Checked before anything is created: an unrecognized value used to
    # configure nothing at all, silently.
    if tracking_source and tracking_source not in TRACKING_SOURCES:
        typer.echo(
            f"Unknown --tracking-source '{tracking_source}'. mldna refuses rather "
            "than configuring no tracking source at all, which would leave the "
            "project on local tracking without saying so. Fix: re-run with "
            f"--tracking-source {' | '.join(TRACKING_SOURCES)}.",
            err=True,
        )
        raise typer.Exit(1)

    # dir has no mldna.yaml of its own, but an ancestor is already a project:
    # scaffolding here would create a confusing second project on top of it
    # (D#57), so refuse instead of silently nesting one.
    if not (Path(dir) / "mldna.yaml").exists() and find_parent_project(dir):
        typer.echo(missing_project_hint(dir), err=True)
        raise typer.Exit(1)

    azure_flag_given = bool(
        azure or subscription_id or resource_group or workspace or tracking_uri
        or auth_mode or tenant_id or client_id or client_secret
    )
    mlflow_flag_given = bool(mlflow_uri)
    filesystem_flag_given = bool(filesystem_path)

    if consumer:
        created = scaffold_consumer(dir)
        for path in created:
            typer.echo(f"created {path}")
        typer.echo("Initialized consumer project.")

        if mlflow_flag_given and not has_mlflow_config(dir):
            _configure_mlflow_tracking(dir, mlflow_uri)
        elif mlflow_flag_given:
            typer.echo("mldna.yaml already has an mlflow: section - not re-prompting.")

        if filesystem_flag_given and not has_filesystem_config(dir):
            _configure_filesystem_tracking(dir, filesystem_path)
        elif filesystem_flag_given:
            typer.echo("mldna.yaml already has a filesystem: section - not re-prompting.")

        if azure_flag_given and not has_azure_config(dir):
            _configure_azure_tracking(
                dir, auth_mode, tracking_uri, tenant_id, client_id, client_secret,
                subscription_id, resource_group, workspace,
            )
            typer.echo(
                "If this machine doesn't have az login access, run "
                "`mldna configure aml-auth` to set a shared secret."
            )
        elif azure_flag_given:
            typer.echo("mldna.yaml already has an azure: section - not re-prompting.")
        return

    if model_name is None:
        typer.echo("model_name is required unless --consumer is set.", err=True)
        raise typer.Exit(1)

    created = scaffold_producer(dir, model_name, namespace=namespace)
    for path in created:
        typer.echo(f"created {path}")
    typer.echo(f"Initialized producer project '{model_name}'.")
    typer.echo(f"Namespace: {namespace or ProjectConfig.load(dir).namespace} (edit project.namespace in mldna.yaml to change it).")

    configured_azure = False
    if has_azure_config(dir) or has_mlflow_config(dir) or has_filesystem_config(dir):
        typer.echo(
            "mldna.yaml already has a tracking source configured (azure:, "
            "mlflow: or filesystem: section) - not re-prompting."
        )
    else:
        if azure_flag_given:
            tracking_source = "azure"
        elif mlflow_flag_given:
            tracking_source = "mlflow"
        elif filesystem_flag_given:
            tracking_source = "filesystem"
        elif not tracking_source:
            # Filesystem is appended rather than inserted so the existing
            # menu numbers keep meaning what they meant.
            tracking_source = list(TRACKING_SOURCES)[
                _choose(
                    "Select MLflow tracking source:",
                    [
                        "Local", "Remote MLflow server (URL)", "Azure ML",
                        "Shared filesystem store (existing path)",
                    ],
                )
            ]
        if tracking_source == "mlflow":
            _configure_mlflow_tracking(dir, mlflow_uri)
        elif tracking_source == "filesystem":
            _configure_filesystem_tracking(dir, filesystem_path)
        elif tracking_source == "azure":
            _configure_azure_tracking(
                dir, auth_mode, tracking_uri, tenant_id, client_id, client_secret,
                subscription_id, resource_group, workspace,
            )
            configured_azure = True

    configured_dvc_azure = False
    if has_dvc_config(dir):
        typer.echo("mldna.yaml already has a dvc: section - not re-prompting.")
    else:
        configured_dvc_azure = _configure_dvc(
            dir, model_name, dvc_backend, dvc_local_path,
            dvc_account, dvc_container, dvc_prefix,
            blob_auth_mode, connection_string,
        )

    if configured_azure or configured_dvc_azure:
        typer.echo(
            "If this machine doesn't have az login access, run "
            "`mldna configure aml-auth` and/or `mldna configure blob-auth` to set "
            "shared secrets."
        )


@app.command()
def stage(
    dir: str = typer.Option(".", "--dir", help="Target project directory"),
) -> None:
    """Prepare a push: dvc add, then git add. Commits nothing."""
    try:
        lines, unstaged, in_history = core_stage(dir)
    except PushError as e:
        typer.echo(f"stage failed: {e}", err=True)
        raise typer.Exit(1)

    for path in unstaged:
        typer.echo(f"Unstaged (now ignored): {path}")
    for path in in_history:
        typer.echo(
            f"Note: {path} is ignored but already in git history, and .gitignore alone "
            "cannot remove it. If it held a secret, rotate the secret."
        )
    if lines:
        typer.echo("Staged files:")
        for line in lines:
            typer.echo(f"  {line}")
    else:
        typer.echo("Nothing staged.")
    typer.echo(
        "Review them in your editor's source control view. Add anything unwanted to "
        ".gitignore, then re-run `mldna stage`. When it looks right, run "
        "`mldna push -m \"<message>\"`."
    )


@app.command()
def push(
    message: str = typer.Option(None, "--message", "-m", help="Commit message"),
    dir: str = typer.Option(".", "--dir", help="Target project directory"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Commit without asking for confirmation"),
) -> None:
    """Commit what `mldna stage` staged, then dvc push and git push."""
    try:
        check_something_to_push(dir)
    except PushError as e:
        typer.echo(f"push failed: {e}", err=True)
        raise typer.Exit(1)

    if message is None:
        if not _stdin_is_interactive():
            typer.echo(
                "No commit message given (-m) and stdin isn't interactive, so "
                "mldna can't prompt for one. Fix: pass -m \"<message>\".",
                err=True,
            )
            raise typer.Exit(1)
        message = typer.prompt("Commit message")

    def confirm(msg):
        body, _, question = msg.rpartition("\n\n")
        typer.echo(body)
        if yes:
            return True
        if not _stdin_is_interactive():
            raise PushError(
                "stdin isn't interactive, so mldna can't ask before committing. A pushed "
                "commit is permanent, so push always asks. Fix: pass --yes."
            )
        return typer.confirm(question, default=False)

    try:
        result = core_push(dir, message=message, confirm=confirm)
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
            if len(run_id) < 4:
                typer.echo(
                    f"--run-id {run_id} is too short: at least 4 characters are needed to "
                    "identify a run. Fix: pass the id shown by `mldna runs`.", err=True
                )
                raise typer.Exit(1)
            matches = [c for c in candidates if c.info.run_id.startswith(run_id)]
            if len(matches) == 1:
                return matches[0]
            problem = "matches several of" if matches else "is not among"
            typer.echo(f"--run-id {run_id} {problem} the runs published from this commit:", err=True)
            for c in matches or candidates:
                typer.echo(
                    f"  {c.info.run_id[:8]}  v{c.data.tags.get('registry_version')}  "
                    f"published {c.data.tags.get('published_at', '?')}", err=True
                )
            typer.echo(
                "Fix: re-run with --run-id set to enough characters of one id above.", err=True
            )
            raise typer.Exit(1)
        if not _stdin_is_interactive():
            typer.echo(
                f"{len(candidates)} published runs match this commit and stdin "
                "isn't interactive, so mldna can't prompt to choose. Code and "
                "data ARE already synced - only the model bundle was not "
                "pulled. The candidates:", err=True
            )
            for c in candidates:
                typer.echo(
                    f"  {c.info.run_id[:8]}  v{c.data.tags.get('registry_version')}  "
                    f"published {c.data.tags.get('published_at', '?')}", err=True
                )
            typer.echo(
                "Fix: re-run with --run-id <id> using one id above; it will "
                "finish the bundle step.", err=True
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
            f"(v{result['registry_version']}) into {result['dest']}/."
        )
    else:
        typer.echo("No published model found for this commit.")


@app.command()
def publish(
    pkl_path: str,
    formats: list[str] = typer.Option([], "--formats", help="Converted format directories"),
    dir: str = typer.Option(".", "--dir", help="Target project directory"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    commit: str = typer.Option(
        None, "--commit",
        help="Choose among runs whose model hash matches but whose lineage differs, by git commit",
    ),
) -> None:
    """Register a model version from a tracked run, resolved by content hash."""
    confirm = (lambda summary: True) if yes else (lambda summary: typer.confirm("Proceed?"))

    def _select(candidates):
        if not _stdin_is_interactive():
            fix = (
                f"--commit {commit} is not enough: these runs share it but cite "
                "different data. Fix: re-run publish in an interactive terminal "
                "and choose one of:"
                if commit else
                f"Fix: re-run as `mldna publish {pkl_path} --commit <sha>` with one of:"
            )
            typer.echo(
                f"publish failed: {len(candidates)} runs match this file's hash with "
                "differing lineage and stdin isn't interactive, so mldna can't "
                f"prompt to choose - it never guesses. {fix}",
                err=True,
            )
            for c in candidates:
                typer.echo(f"  {describe_candidate(c)}", err=True)
            raise typer.Exit(1)
        index = _choose(
            "Multiple runs match this file's hash with differing lineage:",
            [describe_candidate(c) for c in candidates],
        )
        return candidates[index]

    # D#77: a differing origin needs a person even under --yes.
    confirm_origin = (
        (lambda warning: typer.confirm("Publish to this registry model anyway?"))
        if _stdin_is_interactive() else None
    )

    try:
        result = core_publish(pkl_path, workspace_dir=dir, format_dirs=formats,
                               confirm=confirm, commit=commit, select=_select,
                               confirm_origin=confirm_origin)
    except PUBLISH_ERRORS as e:
        typer.echo(f"publish failed: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Published {result['model_name']} v{result['version']} from run {result['run_id']}.")
    typer.echo(
        "Consumers pin it in models.lock.yaml:\n"
        f"  {result['model_name']}:\n    id: {result['model_id']}\n"
        f"    version: {result['version']}\n    format: pkl\n    dest: ./models/<name>"
    )


@app.command("id")
def model_id(
    dir: str = typer.Option(".", "--dir", help="Target project directory"),
    new: bool = typer.Option(False, "--new", help="Replace the id (a project copied from another)"),
) -> None:
    """Give this project its unique model id in mldna.yaml (idempotent)."""
    if not (Path(dir) / "mldna.yaml").exists():
        typer.echo(missing_project_hint(dir), err=True)
        raise typer.Exit(1)
    value, created = ensure_model_id(dir, new=new)
    if new:
        typer.echo(
            f"Wrote a new project.model_id: {value} to mldna.yaml. The registry model "
            "published under the old id will now refuse this project - also change "
            "project.namespace or project.model_name so it publishes to its own "
            "model. Commit mldna.yaml."
        )
    elif created:
        typer.echo(f"Wrote project.model_id: {value} to mldna.yaml. Commit mldna.yaml.")
    else:
        typer.echo(f"project.model_id is already {value}.")


@app.command()
def attach(
    dir: str = typer.Option(".", "--dir", help="Target project directory"),
    run_id: str = typer.Option(None, "--run-id", help="Attach this earlier run (see .mldna/runs/) instead of the latest"),
) -> None:
    """Recover a run left RUNNING by a network failure at run exit:
    re-sends .mldna/last_run.json's model hashes and terminates the run,
    labeled `attached` since its completion was recorded after the fact."""
    try:
        result = core_attach(dir, run_id=run_id)
    except ATTACH_ERRORS as e:
        typer.echo(f"attach failed: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(
        f"Attached run {result['run_id']} "
        f"({result['checksums_written']} checksum tag(s) written)."
    )


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
        if r["skipped"]:
            status = "up to date, skipped"
        elif r["repaired"]:
            status = "repaired (on-disk copy failed verification)"
        else:
            status = "fetched"
        typer.echo(f"{r['model_name']} v{r['version']}: {status}")


KEY_METRIC_WIDTH = 14


def _format_key_metric(value):
    """Six significant digits: fits the column, still separates close runs."""
    text = f"{value:.6g}" if isinstance(value, float) else str(value)
    if len(text) > KEY_METRIC_WIDTH - 1:
        text = text[: KEY_METRIC_WIDTH - 2] + "~"
    return text


@app.command()
def ui(
    dir: str = typer.Option(".", "--dir", help="Target project directory"),
    port: int = typer.Option(8765, "--port", help="Local port to serve on"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open a browser tab"),
) -> None:
    """Read-only local web page of this project's runs, code, data and models."""
    try:
        from mldna.ui.server import serve
    except ImportError as e:
        if e.name not in ("fastapi", "uvicorn", "starlette"):
            raise
        typer.echo(
            "mldna ui needs the optional UI packages, which are not installed. "
            'Fix: pip install "mldna-lineage[ui]"',
            err=True,
        )
        raise typer.Exit(1)
    from mldna.ui.data import UiDataError

    try:
        serve(dir, port=port, open_browser=not no_browser)
    except (UiDataError, TrackingUriError) as e:
        typer.echo(f"ui failed: {e}", err=True)
        raise typer.Exit(1)


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

    typer.echo(
        f"{'RUN ID':<10}{'STARTED':<18}{'COMMIT':<10}"
        f"{'KEY METRIC':<{KEY_METRIC_WIDTH}}{'DIRTY':<8}SAVE_CONTEXT"
    )
    for r in rows:
        typer.echo(
            f"{r['run_id']:<10}{r['started']:<18}{r['git_commit']:<10}"
            f"{_format_key_metric(r['key_metric']):<{KEY_METRIC_WIDTH}}"
            f"{r['dirty']:<8}{r['save_context']}"
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
        typer.echo(f"Saved to .mldna/{aml_auth.AUTH_FILE}")

    try:
        _, mode = aml_auth.resolve_aml_credential(dir)
    except ImportError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    typer.echo(f"Active AML auth mode: {mode}")


@configure_app.command("blob-auth")
def configure_blob_auth(
    connection_string: str = typer.Option(
        None, "--connection-string",
        help=f"Azure Blob connection string; or export {blobauth.ENV_VAR} instead of storing one",
    ),
    remote: str = typer.Option(None, "--remote", help="DVC remote name (defaults to the project's default remote)"),
    dir: str = typer.Option(".", "--dir", help="Target project directory"),
    clear: bool = typer.Option(False, "--clear"),
) -> None:
    """Store (or clear) a shared connection string for the DVC blob remote."""
    remote_name = remote or blobauth.get_default_remote(dir)
    if not remote_name:
        typer.echo("No DVC remote configured. Run `dvc remote add -d <name> <url>` first.", err=True)
        raise typer.Exit(1)

    # --clear runs on any remote: unsetting a key is harmless and idempotent,
    # and it is how a project cleans up a connection string left behind by an
    # earlier azure:// url. Writing one is scheme-checked below.
    if clear:
        blobauth.clear(dir, remote_name)
        typer.echo(f"Cleared local connection string for remote '{remote_name}'.")
        return

    remote_url = blobauth.get_remote_url(dir, remote_name)
    if not blobauth.is_azure_remote(remote_url):
        typer.echo(_non_azure_remote_message(remote_name, remote_url), err=True)
        raise typer.Exit(1)

    if connection_string is None:
        connection_string = typer.prompt("Connection string", hide_input=True)
    blobauth.set_connection_string(dir, remote_name, connection_string)
    typer.echo(f"Saved connection string for remote '{remote_name}' to .dvc/config.local")


@workspace_app.command("new")
def workspace_new(
    name: str,
    dir: str = typer.Option(".", "--dir", help="Target project directory"),
) -> None:
    """Create a parallel working copy that shares this repo's DVC cache."""
    try:
        path = ws.new(dir, name)
    except ws.WorkspaceError as e:
        typer.echo(f"workspace new failed: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Created workspace '{name}' at {path} (shares the DVC cache).")


@workspace_app.command("list")
def workspace_list(
    dir: str = typer.Option(".", "--dir", help="Target project directory"),
) -> None:
    """List this repo's working copies."""
    try:
        rows = ws.list_workspaces(dir)
    except ws.WorkspaceError as e:
        typer.echo(f"workspace list failed: {e}", err=True)
        raise typer.Exit(1)
    for name, path in rows:
        typer.echo(f"{name:<16}{path}")


@workspace_app.command("remove")
def workspace_remove(
    name: str,
    dir: str = typer.Option(".", "--dir", help="Target project directory"),
    yes: bool = typer.Option(False, "--yes", help="Delete gitignored models/mlruns in it without asking"),
) -> None:
    """Remove a parallel working copy; refuses while a run is live in it."""

    def confirm(msg):
        if yes:
            return True
        if not _stdin_is_interactive():
            return False
        typer.echo(msg)
        return typer.confirm("Delete them and remove the workspace?")

    try:
        stale = ws.remove(dir, name, confirm)
    except ws.WorkspaceError as e:
        typer.echo(f"workspace remove failed: {e}", err=True)
        raise typer.Exit(1)
    if stale:
        typer.echo(f"Note: '{name}' had a stale run lock (its process is gone); removed anyway.")
    typer.echo(f"Removed workspace '{name}'.")


@app.command()
def doctor(
    dir: str = typer.Option(".", "--dir", help="Target project directory"),
    timeout: str = typer.Option(None, "--timeout", help="Seconds to wait for the MLflow server, e.g. 60 or 60s (default 10)"),
) -> None:
    """Check tool availability, auth, git remote, gitignore, and large tracked files."""
    try:
        seconds = parse_timeout(timeout) if timeout is not None else None
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2)
    results = run_doctor(dir, seconds)
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
