import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from mldna.core import discovery, hashing, tags as tagschema


class ResolveError(Exception):
    pass


class AmbiguousRunError(ResolveError):
    def __init__(self, message, candidates):
        super().__init__(message)
        self.candidates = candidates


@dataclass
class ResolvedRun:
    run_id: str
    relpath: str
    auto_resolved_note: str = None


def _matching_relpath(run, target_hash):
    for tag_name, tag_value in run.data.tags.items():
        if not tag_name.startswith("model_sha256__"):
            continue
        sha256, _, relpath = tag_value.partition(":")
        if sha256 == target_hash:
            return relpath
    return None


def describe_candidate(run):
    tags = run.data.tags
    started = datetime.fromtimestamp(run.info.start_time / 1000, timezone.utc)
    data = ", ".join(
        f"{out}={md5[:8]}"
        for out, md5 in sorted(tagschema.data_hashes_from_tags(tags).items())
    )
    return (
        f"run {run.info.run_id[:8]}  commit {tags.get('git_commit', '?')[:8]}  "
        f"started {started:%Y-%m-%d %H:%M} UTC  user {tags.get('user', '?')}  "
        f"data {data or 'none'}"
    )


def _local_record_run_id(workspace_dir, target_hash):
    if workspace_dir is None:
        return None
    for record_path in sorted((Path(workspace_dir) / ".mldna" / "runs").glob("*.json")):
        try:
            record = json.loads(record_path.read_text())
            if target_hash in record["checksums"].values():
                return record["run_id"]
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return None


def resolve_run(experiment_name, pkl_path, scan_limit=200, commit=None, select=None,
                workspace_dir=None):
    target_hash = hashing.sha256_file(pkl_path)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ResolveError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        [experiment.experiment_id], order_by=["start_time DESC"], max_results=scan_limit
    )
    # D#51: a full page back (len(runs) == scan_limit) means older runs were
    # never read - the scan may simply not have reached far enough, so a
    # zero-match result here is NOT proof the file was never tracked. Only a
    # short page (fewer runs exist than the bound) licenses that claim.
    truncated = len(runs) >= scan_limit

    matches = []
    for run in runs:
        relpath = _matching_relpath(run, target_hash)
        if relpath:
            matches.append((run, relpath))

    if not matches:
        local_run_id = _local_record_run_id(workspace_dir, target_hash)
        if local_run_id:
            raise ResolveError(
                f"{pkl_path} was saved inside run {local_run_id} on this machine, "
                f"but the tracking server has no record of that (the run likely "
                f"lost its connection when it ended). Fix: run "
                f"`mldna attach --run-id {local_run_id}`, then publish again."
            )
        if truncated:
            raise ResolveError(
                f"{pkl_path} is not in any run recorded on this machine, and the "
                f"scan of '{experiment_name}' stopped at {scan_limit} runs without "
                f"a match - older runs were never examined. Fix: if it was trained "
                f"on another machine, raise tracking.scan_limit in mldna.yaml above "
                f"{scan_limit}; otherwise save it inside `with mldna.track_run():` "
                f"and re-train."
            )
        # If the file itself sits in a directory the scan skips, that is almost
        # certainly why it was never tracked - say so instead of implying the
        # user never used track_run at all (D#54).
        skipped = [
            d for d in discovery.DEFAULT_EXCLUDES
            if d.rstrip("/") in Path(pkl_path).parts
        ]
        if skipped:
            raise ResolveError(
                f"No tracked run matches the SHA-256 of {pkl_path}, and that path "
                f"is inside '{skipped[0]}', which mldna never scans - so a model "
                f"saved there is never recorded. Fix: save models under models/ and "
                f"re-train, so the run sees the file saved inside a tracked run."
            )
        raise ResolveError(
            f"No tracked run matches the SHA-256 of {pkl_path}. The file was "
            f"never saved inside a tracked context, or was modified after saving. "
            f"Fix: publish a file that was saved inside a `with mldna.track_run():` block. "
            f"If that run ended with a warning that it could not reach the tracking "
            f"server, run `mldna attach` (or `mldna attach --run-id <id>` for an "
            f"earlier run saved in .mldna/runs/) and publish again."
        )
        # TODO (Phase 0.2): track_save support - uncomment mention when added
        # f"never saved inside a tracked context (track_run/track_save), "

    dirty = [pair for pair in matches if pair[0].data.tags.get("dirty") == "true"]
    matches = [pair for pair in matches if pair[0].data.tags.get("dirty") != "true"]

    if not matches:
        dirty_ids = ", ".join(r.info.run_id for r, _ in dirty)
        raise ResolveError(
            f"Every run matching the SHA-256 of {pkl_path} is tagged dirty=true "
            f"and can never be published ({dirty_ids}). A dirty "
            f"run's recorded commit does not describe the code that produced the "
            f"file, so its lineage can never be made true. Fix: commit or stash your "
            f"changes, re-train inside `with mldna.track_run():` on a clean tree, "
            f"then publish the model that run saves."
        )

    if len(matches) == 1:
        run, relpath = matches[0]
        return ResolvedRun(run_id=run.info.run_id, relpath=relpath)

    # The data half is every out, sorted by out path: order-insensitive (tag
    # indices differ between runs) and complete (a missing or extra out, or the
    # same hash under a different out path, reads as different lineage, so D#5's
    # never-guess rule still only auto-resolves genuinely identical data).
    lineages = {
        (
            r.data.tags.get("git_commit"),
            tuple(sorted(tagschema.data_hashes_from_tags(r.data.tags).items())),
        )
        for r, _ in matches
    }
    if len(lineages) == 1:
        run, relpath = min(matches, key=lambda pair: pair[0].info.start_time)
        return ResolvedRun(
            run_id=run.info.run_id,
            relpath=relpath,
            auto_resolved_note=(
                f"{len(matches)} runs matched this file's hash, all with identical "
                f"lineage; auto-picked the earliest ({run.info.run_id})."
            ),
        )

    # D#64: every candidate already hash-matched the file, so each lineage is
    # true of these bytes - a human only picks WHICH one is recorded, and the
    # pick is labelled as a human choice wherever the resolution is shown.
    candidates = [r for r, _ in matches]
    relpaths = {r.info.run_id: relpath for r, relpath in matches}
    listing = "\n".join(f"  {describe_candidate(r)}" for r in candidates)
    narrowed = ""
    fix = (
        f"Fix: `mldna publish {pkl_path} --commit <sha>` (Python: "
        f"commit=\"<sha>\"), or run publish in an interactive terminal to "
        f"choose from this list."
    )
    chosen = None

    if commit is not None:
        if len(commit) < 7:
            raise ResolveError(
                f"--commit {commit} is too short: at least 7 characters are "
                f"required so a prefix cannot match a commit by accident. "
                f"Candidates:\n{listing}\n"
                f"Fix: `mldna publish {pkl_path} --commit <sha>` with one of the "
                f"commits above."
            )
        picked = [r for r in candidates if r.data.tags.get("git_commit", "").startswith(commit)]
        if not picked:
            raise ResolveError(
                f"--commit {commit} matches none of the {len(candidates)} runs "
                f"whose model hash matches {pkl_path}; it can only choose among "
                f"them. Candidates:\n{listing}\n"
                f"Fix: `mldna publish {pkl_path} --commit <sha>` with one of the "
                f"commits above."
            )
        if len(picked) == 1:
            chosen, how = picked[0], f"--commit {commit}"
        else:
            candidates = picked
            listing = "\n".join(f"  {describe_candidate(r)}" for r in candidates)
            narrowed = (
                f" --commit {commit} narrowed this to {len(picked)} runs from that "
                f"same commit which cite DIFFERENT data, so commit alone is not enough."
            )
            fix = (
                f"Fix: run `mldna publish {pkl_path}` in an interactive terminal "
                f"to choose from this list."
            )

    if chosen is None and select is not None:
        chosen, how = select(candidates), "interactive choice"
        if chosen.info.run_id not in relpaths:
            raise ResolveError(
                f"The selected run {chosen.info.run_id} is not one of the runs "
                f"whose model hash matches {pkl_path}. Fix: choose one of:\n{listing}"
            )

    if chosen is None:
        raise AmbiguousRunError(
            f"{len(candidates)} runs match this file's hash with DIFFERING "
            f"lineage, and mldna never guesses which one to record.{narrowed} "
            f"Candidates:\n{listing}\n{fix}",
            candidates,
        )

    return ResolvedRun(
        run_id=chosen.info.run_id,
        relpath=relpaths[chosen.info.run_id],
        auto_resolved_note=(
            f"HUMAN-SELECTED lineage: {len(matches)} runs matched this file's hash "
            f"with DIFFERING lineage; {how} selected run {chosen.info.run_id} "
            f"(commit {chosen.data.tags.get('git_commit', '?')[:8]}). The hash "
            f"proves these bytes, not which of those lineages is the intended one."
        ),
    )
