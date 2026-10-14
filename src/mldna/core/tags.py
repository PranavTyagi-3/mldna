GIT_COMMIT = "git_commit"
GIT_BRANCH = "git_branch"
DVC_DATA_HASH = "dvc_data_hash"  # legacy single-out tag; read-only since D#63
DIRTY = "dirty"
USER = "user"
MLDNA_VERSION = "mldna_version"
DOCKER_IMAGE_DIGEST = "docker_image_digest"
SAVE_CONTEXT = "save_context"
ATTACHED = "attached"
ATTACHED_AT = "attached_at"

# Tag on the REGISTERED MODEL (not a run or version): its owning project (D#69).
MODEL_ID = "mldna_model_id"
# Also on the registered model: the normalised git origin it is published from (D#77).
MODEL_ORIGIN = "mldna_origin"
INVOCATION_ARGV = "invocation_argv"
INVOCATION_ENV = "invocation_env"

IN_RUN = "in_run"
POST_RUN = "post_run"

# Model version tags a publish that lost a concurrent race puts on its own version (D#76).
ABANDONED = "abandoned"
ABANDONED_REASON = "abandoned_reason"

# Run tag per published file: published__<first 16 hex of its sha256> = version (D#78).
PUBLISHED_PREFIX = "published__"


def published_tag_name(sha256):
    return f"{PUBLISHED_PREFIX}{sha256[:16]}"


def is_abandoned(version_tags):
    # Fail safe like `recalled` (D#50): present means abandoned unless plainly false.
    value = version_tags.get(ABANDONED)
    return value is not None and value.strip().lower() not in {"false", "no", "0", "off", ""}


def checksum_tag_name(index):
    return f"model_sha256__{index}"


def checksum_tag_value(sha256, relpath):
    return f"{sha256}:{relpath}"


DVC_DATA_HASH_PREFIX = "dvc_data_hash__"

# The out a pre-D#63 run's single bare `dvc_data_hash` described: read_data_hash
# took outs[0] of the root `dataset.dvc`, whose path is `dataset`.
LEGACY_DATA_OUT = "dataset"


def data_hash_tag_name(index):
    return f"{DVC_DATA_HASH_PREFIX}{index}"


def data_hash_tag_value(md5, relpath):
    return f"{md5}:{relpath}"


def data_hashes_from_tags(run_tags):
    """`{out relpath: md5}` from a run's per-out `dvc_data_hash__<n>` tags.

    A run written before D#63 carries one bare `dvc_data_hash` instead; it is
    read as the single `dataset` out it actually described, so old runs stay
    resolvable and publishable and compare equal to a fresh single-out read of
    the same workspace. A legacy run whose out was named something else
    compares unequal and refuses loudly, which is the safe side.
    """
    per_out = {}
    for name, value in run_tags.items():
        if not name.startswith(DVC_DATA_HASH_PREFIX):
            continue
        md5, _, relpath = value.partition(":")
        per_out[relpath] = md5
    if per_out:
        return dict(sorted(per_out.items()))
    legacy = run_tags.get(DVC_DATA_HASH)
    return {LEGACY_DATA_OUT: legacy} if legacy else {}
