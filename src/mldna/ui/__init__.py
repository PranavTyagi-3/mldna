"""Read-only local web UI for mldna (`pip install "mldna-lineage[ui]"`, then `mldna ui`).

Nothing here writes anywhere: the server binds to 127.0.0.1, exposes GET routes
only, and runs a fixed allowlist of read-only git/dvc commands. Nothing outside
this package imports it, so the base install never needs its dependencies.
"""
