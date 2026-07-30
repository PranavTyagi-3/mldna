import subprocess
import sys
import textwrap

# Regression tests for a real bug found live: `pip install mldna` (base, no
# [azure] extra) crashed - not just on import, but with a raw unhandled
# traceback the moment any Azure code path was actually exercised - because
# several places imported azure-ai-ml (an [azure]-extra-only package)
# without catching ImportError. Since D#35 the base dependency is plain
# `dvc`, not `dvc[azure]`, so NO azure.* / azureml.* module is guaranteed
# present on a base install anymore - block the whole namespaces, exactly
# what a real base venv looks like.
BLOCKED_MODULES = ("azure", "azureml")

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
    result = _run_without_azure_extras("import mldna.cli.main; print('OK')")
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_mldna_package_imports_without_azure_extra():
    result = _run_without_azure_extras("import mldna; print('OK')")
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_configure_aml_auth_fails_cleanly_without_azure_extra(tmp_path):
    # Saving the secrets file is pure stdlib and must work; the credential
    # resolution afterward is what needs azure.identity - it must refuse
    # with the pip-install fix, not a raw traceback.
    result = _run_without_azure_extras(f"""
        from typer.testing import CliRunner
        from mldna.cli.main import app
        runner = CliRunner()
        result = runner.invoke(app, [
            "configure", "aml-auth", "--dir", {str(tmp_path)!r},
            "--tenant-id", "t", "--client-id", "c", "--client-secret", "s",
        ])
        print("EXIT_CODE", result.exit_code)
        print("OUTPUT", result.output)
        if result.exception and not isinstance(result.exception, SystemExit):
            raise result.exception
    """)
    assert result.returncode == 0, result.stderr
    assert "EXIT_CODE 1" in result.stdout
    assert 'pip install "mldna-lineage[azure]"' in result.stdout
    assert "Traceback" not in result.stdout
    assert (tmp_path / ".mldna" / "aml_auth.json").exists()


def test_init_dvc_azure_aad_fails_cleanly_without_azure_extra(tmp_path):
    result = _run_without_azure_extras(f"""
        from typer.testing import CliRunner
        from mldna.cli.main import app
        runner = CliRunner()
        result = runner.invoke(app, [
            "init", "test-model", "--dir", {str(tmp_path)!r},
            "--tracking-source", "local", "--dvc-backend", "azure",
            "--blob-auth-mode", "aad", "--dvc-account", "myacct",
        ])
        print("EXIT_CODE", result.exit_code)
        print("OUTPUT", result.output)
        if result.exception and not isinstance(result.exception, SystemExit):
            raise result.exception
    """)
    assert result.returncode == 0, result.stderr
    assert "EXIT_CODE 1" in result.stdout
    assert 'pip install "mldna-lineage[azure]"' in result.stdout
    assert "Traceback" not in result.stdout


def test_init_azure_ad_discovery_fails_cleanly_without_azure_extra(tmp_path):
    # Real bug: choosing the Azure AD path in `mldna init` without the
    # [azure] extra installed raised a raw ModuleNotFoundError traceback
    # instead of a clean, actionable error.
    result = _run_without_azure_extras(f"""
        from typer.testing import CliRunner
        from mldna.cli.main import app
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
    assert 'pip install "mldna-lineage[azure]"' in result.stdout
    assert "Traceback" not in result.stdout


def test_bootstrap_path_fails_cleanly_without_azure_extra(tmp_path):
    # Same gap, the other call site: a project already configured with
    # subscription_id/resource_group/workspace (e.g. cloned from a teammate)
    # must raise a clean TrackingUriError, not an unhandled ImportError, when
    # the [azure] extra isn't installed on this machine.
    (tmp_path / "mldna.yaml").write_text(
        "project:\n  model_name: test-model\n  repo_role: producer\n"
        "azure:\n  subscription_id: s\n  resource_group: rg\n  workspace: ws\n"
    )
    result = _run_without_azure_extras(f"""
        from mldna.trackinguri import TrackingUriError, resolve_and_set_tracking_uri
        try:
            resolve_and_set_tracking_uri({str(tmp_path)!r})
        except TrackingUriError as e:
            print("CLEAN_ERROR", e)
    """)
    assert result.returncode == 0, result.stderr
    assert "CLEAN_ERROR" in result.stdout
    assert 'pip install "mldna-lineage[azure]"' in result.stdout
    assert "Traceback" not in result.stdout


def test_tracking_uri_path_fails_cleanly_without_azure_extra(tmp_path):
    # The Service Principal / tracking_uri shortcut never touches azure-ai-ml
    # at all (that's its whole point - see D#29) so it was fully unguarded:
    # mlflow.set_tracking_uri() itself doesn't fail, and the real crash only
    # happened on the *first actual mlflow call*, deep inside mlflow's own
    # store registry, as an UnsupportedModelRegistryStoreURIException that
    # never mentions mldna or the fix.
    (tmp_path / "mldna.yaml").write_text(
        "project:\n  model_name: test-model\n  repo_role: producer\n"
        "azure:\n  tracking_uri: azureml://fake-endpoint\n"
    )
    result = _run_without_azure_extras(f"""
        from mldna.trackinguri import TrackingUriError, resolve_and_set_tracking_uri
        try:
            resolve_and_set_tracking_uri({str(tmp_path)!r})
        except TrackingUriError as e:
            print("CLEAN_ERROR", e)
    """)
    assert result.returncode == 0, result.stderr
    assert "CLEAN_ERROR" in result.stdout
    assert 'pip install "mldna-lineage[azure]"' in result.stdout
    assert "Traceback" not in result.stdout
