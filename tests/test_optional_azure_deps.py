import subprocess
import sys
import textwrap

# Regression tests for a real bug found live: `pip install mlops` (base, no
# [azure] extra) crashed - not just on import, but with a raw unhandled
# traceback the moment any Azure code path was actually exercised - because
# several places imported azure-ai-ml (an [azure]-extra-only package)
# without catching ImportError. `azure.identity` is fine to import
# unconditionally: it's a real transitive dependency of the base
# `dvc[azure]` requirement, not extra-only.
BLOCKED_MODULES = ("azure.ai.ml", "azureml.mlflow")

_BLOCKER_PREAMBLE = textwrap.dedent(f"""
    import sys
    class _Blocker:
        def find_spec(self, name, path=None, target=None):
            if name in {BLOCKED_MODULES!r} or any(
                name.startswith(m + ".") for m in {BLOCKED_MODULES!r}
            ):
                raise ImportError(f"{{name}} is [azure]-extra only, blocked for this test")
            return None
    sys.meta_path.insert(0, _Blocker())
""")


def _run_without_azure_extras(body):
    result = subprocess.run(
        [sys.executable, "-c", _BLOCKER_PREAMBLE + textwrap.dedent(body)],
        capture_output=True, text=True,
    )
    return result


def test_cli_main_imports_without_azure_extra():
    result = _run_without_azure_extras("import mlops.cli.main; print('OK')")
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_mlops_package_imports_without_azure_extra():
    result = _run_without_azure_extras("import mlops; print('OK')")
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_init_azure_ad_discovery_fails_cleanly_without_azure_extra(tmp_path):
    # Real bug: choosing the Azure AD path in `mlops init` without the
    # [azure] extra installed raised a raw ModuleNotFoundError traceback
    # instead of a clean, actionable error.
    result = _run_without_azure_extras(f"""
        from typer.testing import CliRunner
        from mlops.cli.main import app
        runner = CliRunner()
        result = runner.invoke(app, [
            "init", "test-model", "--dir", {str(tmp_path)!r},
            "--tracking-source", "azure", "--auth-mode", "aad",
        ], input="sub-123\\n")
        print("EXIT_CODE", result.exit_code)
        print("OUTPUT", result.output)
        if result.exception and not isinstance(result.exception, SystemExit):
            raise result.exception
    """)
    assert result.returncode == 0, result.stderr
    assert "EXIT_CODE 1" in result.stdout
    assert 'pip install "mlops[azure]"' in result.stdout
    assert "Traceback" not in result.stdout


def test_bootstrap_path_fails_cleanly_without_azure_extra(tmp_path):
    # Same gap, the other call site: a project already configured with
    # subscription_id/resource_group/workspace (e.g. cloned from a teammate)
    # must raise a clean TrackingUriError, not an unhandled ImportError, when
    # the [azure] extra isn't installed on this machine.
    (tmp_path / "mlops.yaml").write_text(
        "project:\n  model_name: test-model\n  repo_role: producer\n"
        "azure:\n  subscription_id: s\n  resource_group: rg\n  workspace: ws\n"
    )
    result = _run_without_azure_extras(f"""
        from mlops.trackinguri import TrackingUriError, resolve_and_set_tracking_uri
        try:
            resolve_and_set_tracking_uri({str(tmp_path)!r})
        except TrackingUriError as e:
            print("CLEAN_ERROR", e)
    """)
    assert result.returncode == 0, result.stderr
    assert "CLEAN_ERROR" in result.stdout
    assert 'pip install "mlops[azure]"' in result.stdout
    assert "Traceback" not in result.stdout


def test_tracking_uri_path_fails_cleanly_without_azure_extra(tmp_path):
    # The Service Principal / tracking_uri shortcut never touches azure-ai-ml
    # at all (that's its whole point - see D#29) so it was fully unguarded:
    # mlflow.set_tracking_uri() itself doesn't fail, and the real crash only
    # happened on the *first actual mlflow call*, deep inside mlflow's own
    # store registry, as an UnsupportedModelRegistryStoreURIException that
    # never mentions mlops or the fix.
    (tmp_path / "mlops.yaml").write_text(
        "project:\n  model_name: test-model\n  repo_role: producer\n"
        "azure:\n  tracking_uri: azureml://fake-endpoint\n"
    )
    result = _run_without_azure_extras(f"""
        from mlops.trackinguri import TrackingUriError, resolve_and_set_tracking_uri
        try:
            resolve_and_set_tracking_uri({str(tmp_path)!r})
        except TrackingUriError as e:
            print("CLEAN_ERROR", e)
    """)
    assert result.returncode == 0, result.stderr
    assert "CLEAN_ERROR" in result.stdout
    assert 'pip install "mlops[azure]"' in result.stdout
    assert "Traceback" not in result.stdout
