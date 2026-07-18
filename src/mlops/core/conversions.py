from pathlib import Path

from mlops.core import hashing


class ConversionError(Exception):
    pass


def build_entries(hashed_files, source_pkl_sha, at):
    return {
        sha: {"relpath": relpath, "source_pkl_sha256": source_pkl_sha, "at": at}
        for relpath, sha in hashed_files.items()
    }


def verify_bundle(conversions, format_dirs, source_pkl_sha):
    for format_dir in format_dirs:
        format_dir = Path(format_dir)
        for file in format_dir.rglob("*"):
            if not file.is_file():
                continue
            sha = hashing.sha256_file(file)
            entry = conversions.get(sha)
            if entry is None:
                raise ConversionError(
                    f"{file} is not recorded as a tracked conversion. Every file "
                    f"in a published bundle must be produced inside "
                    f"mlops.track_convert(...) so its provenance chains to the pkl. "
                    f"Fix: re-run the conversion inside a track_convert block."
                )
            if entry["source_pkl_sha256"] != source_pkl_sha:
                raise ConversionError(
                    f"{file} was converted from a different pkl "
                    f"(source {entry['source_pkl_sha256'][:12]}..., publishing "
                    f"{source_pkl_sha[:12]}...). The pkl was likely retrained or "
                    f"modified after this conversion. Fix: re-run the conversion "
                    f"inside a track_convert block against the current pkl."
                )
