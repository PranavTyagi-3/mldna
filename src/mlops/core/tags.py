GIT_COMMIT = "git_commit"
GIT_BRANCH = "git_branch"
DVC_DATA_HASH = "dvc_data_hash"
DIRTY = "dirty"
USER = "user"
MLOPS_VERSION = "mlops_version"
DOCKER_IMAGE_DIGEST = "docker_image_digest"
SAVE_CONTEXT = "save_context"

IN_RUN = "in_run"
POST_RUN = "post_run"


def checksum_tag_name(index):
    return f"model_sha256__{index}"


def checksum_tag_value(sha256, relpath):
    return f"{sha256}:{relpath}"
