import hashlib
import itertools
import json

import mlflow
import pytest

from mldna.core import tags as tagschema
from mldna.core.resolve import AmbiguousRunError, ResolveError, resolve_run

MODEL_BYTES = b"identical model bytes for hash matching"

EXPERIMENT = "project-test-model"
COMMIT_A = "a1" * 20
COMMIT_B = "b2" * 20
DATA = [("dataset", "aaaa1111")]
TWO_OUTS = [("dataset", "aaaa1111"), ("dataset2", "bbbb2222")]

# Runs are written straight to the tracking store with the tags track_run
# would have set, so these tests exercise resolve_run alone. Start times
# increase per run so "earliest" is well defined.
_clock = itertools.count(1_700_000_000_000, 1000)


@pytest.fixture
def client(tmp_path):
    mlflow.set_tracking_uri(f"file:{tmp_path / 'mlruns'}")
    return mlflow.tracking.MlflowClient()


def _run(client, relpath="models/model.pkl", content=MODEL_BYTES, commit=COMMIT_A,
         data=DATA, dirty=False, legacy_data_hash=None):
    experiment = client.get_experiment_by_name(EXPERIMENT)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(EXPERIMENT)
    tags = {
        tagschema.GIT_COMMIT: commit,
        tagschema.USER: "dev@example.com",
        tagschema.checksum_tag_name(0): tagschema.checksum_tag_value(
            hashlib.sha256(content).hexdigest(), relpath
        ),
    }
    for index, (out, md5) in enumerate(data if legacy_data_hash is None else []):
        tags[tagschema.data_hash_tag_name(index)] = tagschema.data_hash_tag_value(md5, out)
    if legacy_data_hash:
        tags[tagschema.DVC_DATA_HASH] = legacy_data_hash
    if dirty:
        tags[tagschema.DIRTY] = "true"
    run = client.create_run(experiment_id, start_time=next(_clock), tags=tags)
    client.set_terminated(run.info.run_id)
    return run.info.run_id


def _candidate(tmp_path, content=MODEL_BYTES):
    candidate = tmp_path / "to_publish.pkl"
    candidate.write_bytes(content)
    return candidate


def test_resolve_finds_single_match(client, tmp_path):
    run_id = _run(client)

    resolved = resolve_run(EXPERIMENT, _candidate(tmp_path))
    assert resolved.run_id == run_id
    assert resolved.relpath == "models/model.pkl"
    assert resolved.auto_resolved_note is None


def test_resolve_zero_match_raises(client, tmp_path):
    _run(client)

    with pytest.raises(ResolveError, match="No tracked run matches"):
        resolve_run(EXPERIMENT, _candidate(tmp_path, b"different bytes"))


def test_resolve_ambiguous_identical_lineage_auto_resolves(client, tmp_path):
    run1_id = _run(client, relpath="models/model_a.pkl")
    _run(client, relpath="models/model_b.pkl")

    resolved = resolve_run(EXPERIMENT, _candidate(tmp_path))
    assert resolved.run_id == run1_id
    assert resolved.relpath == "models/model_a.pkl"
    assert resolved.auto_resolved_note is not None


def test_resolve_ambiguous_differing_lineage_raises(client, tmp_path):
    _run(client, relpath="models/model_a.pkl", commit=COMMIT_A)
    _run(client, relpath="models/model_b.pkl", commit=COMMIT_B)

    with pytest.raises(ResolveError, match="DIFFERING lineage"):
        resolve_run(EXPERIMENT, _candidate(tmp_path))


def test_resolve_refuses_when_any_single_out_hash_differs(client, tmp_path):
    # D#5 never guesses: identical code and one identical out are not identical
    # lineage if a second out differs. Commit stays equal so only the data half
    # of the tie-break can decide.
    _run(client, relpath="models/model_a.pkl", data=TWO_OUTS)
    _run(client, relpath="models/model_b.pkl",
         data=[("dataset", "aaaa1111"), ("dataset2", "deadbeef.dir")])

    with pytest.raises(ResolveError, match="DIFFERING lineage"):
        resolve_run(EXPERIMENT, _candidate(tmp_path))


def test_resolve_auto_resolves_when_per_out_hashes_match_in_any_order(client, tmp_path):
    # Tag indices are per run and carry no meaning, so the comparison is over
    # the SET of (out, hash) pairs - swapping one run's indices must not read
    # as different data (D#63).
    run1_id = _run(client, relpath="models/model_a.pkl", data=TWO_OUTS)
    _run(client, relpath="models/model_b.pkl", data=list(reversed(TWO_OUTS)))

    resolved = resolve_run(EXPERIMENT, _candidate(tmp_path))
    assert resolved.run_id == run1_id
    assert resolved.auto_resolved_note is not None


def test_resolve_treats_a_legacy_single_hash_run_as_the_same_lineage(client, tmp_path):
    # A pre-D#63 run's bare dvc_data_hash described the root `dataset` out, so
    # it must tie-break equal to a per-out run over the same single out rather
    # than manufacturing ambiguity the user cannot resolve.
    run1_id = _run(client, relpath="models/model_a.pkl")
    _run(client, relpath="models/model_b.pkl", legacy_data_hash="aaaa1111")

    resolved = resolve_run(EXPERIMENT, _candidate(tmp_path))
    assert resolved.run_id == run1_id
    assert resolved.auto_resolved_note is not None


def test_resolve_truncated_scan_with_no_match_states_scan_was_truncated(client, tmp_path):
    # scan_limit=1 with one existing (non-matching) run means the returned page
    # equals the bound - the scan cannot tell whether older runs exist, so it
    # must not claim the file was never tracked.
    _run(client, content=b"unrelated bytes")

    with pytest.raises(ResolveError) as excinfo:
        resolve_run(EXPERIMENT, _candidate(tmp_path), scan_limit=1)

    message = str(excinfo.value)
    assert "scan" in message.lower()
    assert "tracking.scan_limit" in message
    assert "never saved" not in message


def _local_record(workspace, run_id, content=MODEL_BYTES):
    runs_dir = workspace / ".mldna" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    checksums = {"models/model.pkl": hashlib.sha256(content).hexdigest()}
    (runs_dir / f"{run_id}.json").write_text(
        json.dumps({"run_id": run_id, "checksums": checksums})
    )


def test_resolve_no_match_with_local_record_advises_attach(client, tmp_path):
    _run(client, content=b"unrelated bytes")
    _local_record(tmp_path, "abc123run")

    with pytest.raises(ResolveError) as excinfo:
        resolve_run(EXPERIMENT, _candidate(tmp_path), scan_limit=1, workspace_dir=tmp_path)

    message = str(excinfo.value)
    assert "mldna attach --run-id abc123run" in message
    assert "tracking.scan_limit" not in message


def test_resolve_truncated_no_local_record_says_not_recorded_on_this_machine(client, tmp_path):
    _run(client, content=b"unrelated bytes")
    _local_record(tmp_path, "otherrun", content=b"other bytes")

    with pytest.raises(ResolveError) as excinfo:
        resolve_run(EXPERIMENT, _candidate(tmp_path), scan_limit=1, workspace_dir=tmp_path)

    message = str(excinfo.value)
    assert "not in any run recorded on this machine" in message
    assert "stopped at 1 runs" in message
    assert "attach" not in message


def test_resolve_malformed_local_record_is_ignored(client, tmp_path):
    _run(client, content=b"unrelated bytes")
    runs_dir = tmp_path / ".mldna" / "runs"
    runs_dir.mkdir(parents=True)
    (runs_dir / "bad.json").write_text("{not json")
    (runs_dir / "list.json").write_text("[]")
    (runs_dir / "nochecksums.json").write_text('{"run_id": "x"}')

    with pytest.raises(ResolveError, match="No tracked run matches"):
        resolve_run(EXPERIMENT, _candidate(tmp_path), workspace_dir=tmp_path)


def test_resolve_exhaustive_scan_with_no_match_keeps_never_tracked_message(client, tmp_path):
    # scan_limit comfortably exceeds the number of runs that exist, so the scan
    # genuinely saw everything and the original wording is still correct.
    _run(client, content=b"unrelated bytes")

    with pytest.raises(ResolveError) as excinfo:
        resolve_run(EXPERIMENT, _candidate(tmp_path), scan_limit=200)

    message = str(excinfo.value)
    assert "No tracked run matches" in message
    assert "tracking.scan_limit" not in message


def test_resolve_all_matches_dirty_raises(client, tmp_path):
    _run(client, dirty=True)

    with pytest.raises(ResolveError, match="dirty=true"):
        resolve_run(EXPERIMENT, _candidate(tmp_path))


def test_resolve_ignores_dirty_match_and_picks_clean_run(client, tmp_path):
    _run(client, relpath="models/model_dirty.pkl", dirty=True)
    clean_run_id = _run(client, relpath="models/model_clean.pkl", commit=COMMIT_B)

    resolved = resolve_run(EXPERIMENT, _candidate(tmp_path))
    assert resolved.run_id == clean_run_id
    assert resolved.relpath == "models/model_clean.pkl"
    assert resolved.auto_resolved_note is None


# --- D#64: differing lineage is resolved by asking, never by guessing ---

def _two_clean_runs_differing_commits(client, tmp_path):
    run_a = _run(client, relpath="models/model_a.pkl", commit=COMMIT_A)
    run_b = _run(client, relpath="models/model_b.pkl", commit=COMMIT_B)
    return (run_a, COMMIT_A), (run_b, COMMIT_B), _candidate(tmp_path)


def test_resolve_differing_lineage_without_help_lists_candidates_and_names_commit(client, tmp_path):
    (run_a, commit_a), (run_b, commit_b), candidate = _two_clean_runs_differing_commits(
        client, tmp_path
    )

    with pytest.raises(AmbiguousRunError) as excinfo:
        resolve_run(EXPERIMENT, candidate)

    message = str(excinfo.value)
    assert commit_a[:8] in message and commit_b[:8] in message
    assert run_a[:8] in message and run_b[:8] in message
    assert "--commit <sha>" in message
    assert 'commit="<sha>"' in message
    assert {r.info.run_id for r in excinfo.value.candidates} == {run_a, run_b}


def test_resolve_commit_too_short_refuses_listing_candidates(client, tmp_path):
    (_, commit_a), (_, commit_b), candidate = _two_clean_runs_differing_commits(
        client, tmp_path
    )

    with pytest.raises(ResolveError, match="too short") as excinfo:
        resolve_run(EXPERIMENT, candidate, commit=commit_a[:6])

    assert commit_a[:8] in str(excinfo.value) and commit_b[:8] in str(excinfo.value)


def test_resolve_commit_matching_no_candidate_refuses_listing_candidates(client, tmp_path):
    (_, commit_a), (_, commit_b), candidate = _two_clean_runs_differing_commits(
        client, tmp_path
    )

    with pytest.raises(ResolveError, match="matches none") as excinfo:
        resolve_run(EXPERIMENT, candidate, commit="0000000")

    assert commit_a[:8] in str(excinfo.value) and commit_b[:8] in str(excinfo.value)


def test_resolve_commit_selects_each_candidate_and_labels_it(client, tmp_path):
    (run_a, commit_a), (run_b, commit_b), candidate = _two_clean_runs_differing_commits(
        client, tmp_path
    )

    for run_id, commit in ((run_a, commit_a), (run_b, commit_b)):
        resolved = resolve_run(EXPERIMENT, candidate, commit=commit[:7])
        assert resolved.run_id == run_id
        assert "HUMAN-SELECTED" in resolved.auto_resolved_note
        assert f"--commit {commit[:7]}" in resolved.auto_resolved_note


def test_resolve_select_callback_chooses_and_labels_it(client, tmp_path):
    (run_a, _), (run_b, _), candidate = _two_clean_runs_differing_commits(client, tmp_path)
    seen = {}

    def select(candidates):
        seen["ids"] = {r.info.run_id for r in candidates}
        return next(r for r in candidates if r.info.run_id == run_a)

    resolved = resolve_run(EXPERIMENT, candidate, select=select)

    assert seen["ids"] == {run_a, run_b}
    assert resolved.run_id == run_a
    assert resolved.relpath == "models/model_a.pkl"
    assert "HUMAN-SELECTED" in resolved.auto_resolved_note
    assert "interactive choice" in resolved.auto_resolved_note


def test_resolve_same_commit_differing_data_stays_ambiguous_under_commit(client, tmp_path):
    _run(client, relpath="models/model_a.pkl", data=TWO_OUTS)
    _run(client, relpath="models/model_b.pkl",
         data=[("dataset", "aaaa1111"), ("dataset2", "deadbeef.dir")])

    with pytest.raises(AmbiguousRunError, match="commit alone is not enough"):
        resolve_run(EXPERIMENT, _candidate(tmp_path), commit=COMMIT_A)


def test_resolve_identical_lineage_auto_resolves_without_asking(client, tmp_path):
    run1_id = _run(client, relpath="models/model_a.pkl")
    _run(client, relpath="models/model_b.pkl")

    def select(candidates):
        raise AssertionError("identical lineage must auto-resolve without asking")

    resolved = resolve_run(EXPERIMENT, _candidate(tmp_path), select=select)
    assert resolved.run_id == run1_id
    assert "identical lineage" in resolved.auto_resolved_note
    assert "HUMAN-SELECTED" not in resolved.auto_resolved_note


def test_resolve_zero_match_in_excluded_dir_says_save_under_models(client, tmp_path):
    _run(client)

    candidate = tmp_path / "outputs" / "model.pkl"
    candidate.parent.mkdir()
    candidate.write_bytes(b"never tracked bytes")

    with pytest.raises(ResolveError) as exc:
        resolve_run(EXPERIMENT, candidate)

    assert "save models under models/ and re-train" in str(exc.value)
    assert "model_globs" not in str(exc.value)
