"""Azure-free core: bundle/manifest logic, lockfile resolution, checksum machinery,
lineage tag schema. mldna.core MUST NOT import anything from azure.* (D#16, CLAUDE.md
invariant #11) — Azure specifics live exclusively in mldna.azure.
"""
