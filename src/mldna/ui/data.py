"""Everything the UI shows, as plain dicts. No web framework in here, so each
function can be called and debugged directly from a Python prompt."""
from pathlib import Path

import mlflow
import yaml
from mlflow.exceptions import MlflowException

from mldna.core import git as core_git
from mldna.core import tags
from mldna.core.config import ProjectConfig, missing_project_hint
from mldna.trackinguri import resolve_and_set_tracking_uri
from mldna.ui import gitread

DEFAULT_LIMIT = 200

_WEB_HOSTS = {
    "github.com": ("/commit/{sha}", "/compare/{old}...{new}"),
    "gitlab.com": ("/-/commit/{sha}", "/-/compare/{old}...{new}"),
    "bitbucket.org": ("/commits/{sha}", None),
}


class UiDataError(Exception):
    pass


def web_url(origin):
    """https URL of a known git host for a normalised origin, else None."""
    if not origin:
        return None
    scheme, sep, rest = origin.partition("://")
    if not sep:
        rest = origin.replace(":", "/", 1)  # git@host:org/repo, userinfo already stripped
    host = rest.split("/", 1)[0]
    return f"https://{rest}" if host in _WEB_HOSTS else None


def _labels(run_tags):
    labels = []
    if run_tags.get(tags.DIRTY) == "true":
        labels.append("dirty")
    if run_tags.get(tags.ATTACHED) == "true":
        labels.append("attached after the fact")
    return labels


def _published(run_tags):
    versions = {v for k, v in run_tags.items() if k.startswith(tags.PUBLISHED_PREFIX)}
    if "registry_version" in run_tags:
        versions.add(run_tags["registry_version"])
    return sorted(versions, key=lambda v: int(v) if v.isdigit() else 0)


def _models(run_tags):
    models = []
    for name, value in run_tags.items():
        if name.startswith("model_sha256__"):
            sha, _, relpath = value.partition(":")
            models.append({"path": relpath, "sha256": sha})
    return sorted(models, key=lambda m: m["path"])


def _pointer_outs(text):
    """{out path: {md5, size, nfiles}} from a .dvc pointer file's text."""
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return {}
    outs = {}
    for out in data.get("outs") or []:
        if isinstance(out, dict) and out.get("path"):
            outs[out["path"]] = {
                "md5": out.get("md5"), "size": out.get("size"), "nfiles": out.get("nfiles"),
            }
    return outs


class UiData:
    def __init__(self, workspace_dir="."):
        self.workspace_dir = Path(workspace_dir).resolve()
        if not (self.workspace_dir / "mldna.yaml").exists():
            raise UiDataError(missing_project_hint(self.workspace_dir))
        self.config = ProjectConfig.load(self.workspace_dir)
        if not self.config.model_name:
            raise UiDataError(
                "mldna ui shows a producer project's runs, and this mldna.yaml has no "
                "project.model_name. Fix: run `mldna ui` inside a producer project."
            )
        self.model_name = self.config.full_model_name()
        self.tracking_uri, self.tracking_mode = resolve_and_set_tracking_uri(self.workspace_dir)
        self.client = mlflow.tracking.MlflowClient()
        self.web = web_url(core_git.origin_url(self.workspace_dir))
        self._subjects = None
        self._code_changes = {}

    # --- links ---

    def commit_link(self, sha):
        if not self.web or not sha:
            return None
        host = self.web.split("/")[2]
        return self.web + _WEB_HOSTS[host][0].format(sha=sha)

    def compare_link(self, old, new):
        if not self.web or not old or not new:
            return None
        pattern = _WEB_HOSTS[self.web.split("/")[2]][1]
        return self.web + pattern.format(old=old, new=new) if pattern else None

    # --- summary and runs ---

    def summary(self):
        return {
            "model_name": self.model_name,
            "tracking_mode": self.tracking_mode,
            "key_metric": self.config.key_metric,
            "remote": self.web,
            "workspace": str(self.workspace_dir),
        }

    def _subject(self, sha):
        if self._subjects is None:
            try:
                out = gitread.git(self.workspace_dir, "log", "--all", "--format=%H%x00%s")
            except gitread.GitReadError:
                out = ""
            self._subjects = dict(line.split("\x00", 1) for line in out.splitlines() if "\x00" in line)
        return self._subjects.get(sha)

    def _code_changed(self, old, new):
        """True when a file other than a DVC pointer changed between the two commits.
        A commit missing from this clone can't be inspected, so a differing one counts."""
        if not old or not new or old == new:
            return False
        if (old, new) not in self._code_changes:
            try:
                files = gitread.changed_files(self.workspace_dir, old, new)
                changed = any(not gitread.is_pointer(f) for f in files)
            except gitread.GitReadError:
                changed = True
            self._code_changes[(old, new)] = changed
        return self._code_changes[(old, new)]

    def _experiment(self):
        return self.client.get_experiment_by_name(self.model_name)

    def _row(self, run):
        run_tags = run.data.tags
        sha = run_tags.get(tags.GIT_COMMIT)
        metric = self.config.key_metric
        return {
            "run_id": run.info.run_id,
            "started": run.info.start_time,
            "status": run.info.status,
            "user": run_tags.get(tags.USER),
            "commit": sha,
            "subject": self._subject(sha) if sha else None,
            "commit_link": self.commit_link(sha),
            "key_metric": run.data.metrics.get(metric) if metric else None,
            "labels": _labels(run_tags),
            "published": _published(run_tags),
            "data": tags.data_hashes_from_tags(run_tags),
        }

    def runs(self, limit=DEFAULT_LIMIT):
        experiment = self._experiment()
        if experiment is None:
            return []
        found = self.client.search_runs(
            [experiment.experiment_id], order_by=["attribute.start_time DESC"], max_results=limit
        )
        rows = [self._row(run) for run in found]
        for row, older in zip(rows, rows[1:] + [None]):
            row["previous_run"] = older["run_id"] if older else None
            row["code_changed"] = bool(older) and self._code_changed(older["commit"], row["commit"])
            row["data_changed"] = bool(older) and row["data"] != older["data"]
        return rows

    def run(self, run_id):
        run = self.client.get_run(run_id)
        row = self._row(run)
        older = self._previous(run)
        run_tags = run.data.tags
        row.update({
            "commit_info": gitread.commit_info(self.workspace_dir, row["commit"]) if row["commit"] else None,
            "previous_run": older.info.run_id if older else None,
            "previous_commit": older.data.tags.get(tags.GIT_COMMIT) if older else None,
            "code_changed": bool(older) and self._code_changed(older.data.tags.get(tags.GIT_COMMIT), row["commit"]),
            "previous_data": tags.data_hashes_from_tags(older.data.tags) if older else None,
            "params": dict(sorted(run.data.params.items())),
            "metrics": dict(sorted(run.data.metrics.items())),
            "models": _models(run_tags),
            "branch": run_tags.get(tags.GIT_BRANCH),
            "invocation": run_tags.get(tags.INVOCATION_ARGV),
            "ended": run.info.end_time,
        })
        return row

    def _previous(self, run):
        found = self.client.search_runs(
            [run.info.experiment_id],
            filter_string=f"attributes.start_time < {run.info.start_time}",
            order_by=["attribute.start_time DESC"], max_results=1,
        )
        return found[0] if found else None

    # --- code ---

    def code_diff(self, old, new):
        for sha in (old, new):
            if not gitread.commit_exists(self.workspace_dir, sha):
                raise UiDataError(
                    f"Commit {sha[:8]} is not in this clone, so its changes cannot be shown. "
                    f"Fix: `git fetch` in {self.workspace_dir}, then reload."
                )
        return {
            "old": old, "new": new,
            "commits": gitread.commits_between(self.workspace_dir, old, new),
            "diff": gitread.diff(self.workspace_dir, old, new),
            "pointers": [f for f in gitread.changed_files(self.workspace_dir, old, new) if gitread.is_pointer(f)],
            "compare_link": self.compare_link(old, new),
        }

    def files(self, sha):
        return gitread.files_at(self.workspace_dir, sha)

    def blame(self, sha, path):
        result = gitread.blame(self.workspace_dir, sha, path)
        for commit in result["commits"].values():
            commit["link"] = self.commit_link(commit["sha"])
        return result

    # --- data ---

    def data_diff(self, old, new):
        """Per-out changes from the .dvc pointers at both commits (works without the data
        itself), plus a file-level list when dvc can produce one from the local cache."""
        pointers = sorted(
            {p for sha in (old, new) for p in gitread.files_at(self.workspace_dir, sha) if p.endswith(".dvc")}
        )
        outs = []
        for pointer in pointers:
            before = _pointer_outs(gitread.show_file(self.workspace_dir, old, pointer) or "")
            after = _pointer_outs(gitread.show_file(self.workspace_dir, new, pointer) or "")
            for path in sorted(set(before) | set(after)):
                a, b = before.get(path), after.get(path)
                if a == b:
                    continue
                outs.append({"pointer": pointer, "path": path, "before": a, "after": b})
        return {"old": old, "new": new, "outs": outs,
                "files": gitread.dvc_diff(self.workspace_dir, old, new) if outs else None}

    # --- models ---

    def models(self):
        try:
            registered = self.client.get_registered_model(self.model_name)
        except MlflowException:
            return {"name": self.model_name, "registered": False, "versions": []}
        versions = self.client.search_model_versions(f"name='{self.model_name}'")
        rows = []
        for mv in sorted(versions, key=lambda v: int(v.version), reverse=True):
            version_tags = mv.tags or {}
            labels = []
            if "recalled" in version_tags:
                labels.append(f"recalled ({version_tags['recalled']})")
            if tags.is_abandoned(version_tags):
                labels.append("abandoned")
            rows.append({
                "version": mv.version,
                "created": mv.creation_timestamp,
                "run_id": version_tags.get("source_run") or mv.run_id,
                "formats": version_tags.get("formats"),
                "published_by": version_tags.get("published_by"),
                "labels": labels,
            })
        model_tags = registered.tags or {}
        return {
            "name": self.model_name, "registered": True, "versions": rows,
            "model_id": model_tags.get(tags.MODEL_ID),
            "origin": model_tags.get(tags.MODEL_ORIGIN),
        }
