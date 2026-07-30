# mldna

**A lineage layer for ML teams who refuse to be told how to train.**

`mldna` is a thin Python package + CLI that answers one question, mechanically
and forever: *which code, which data, and whose hands produced the model
running in production right now?*

It does this without a pipeline DSL, without a required project layout, and
without asking you to change how you train. You keep your scripts, your
notebooks, your favorite framework. `mldna` just makes sure the trail never
goes cold.

---

## The problem

This is the state most ML teams quietly end up in:

- Everyone trains on shared machines, however they like.
- "Dataset versioning" is a folder of files on a blob store, disconnected
  from the code that consumed them.
- Model binaries get copied by hand into wherever downstream apps expect
  them.
- Six months later, nobody can answer *"what produced this model?"* — and
  the honest answer is usually "we're not sure."

Pipeline frameworks (ZenML, Kedro, Metaflow, ...) fix this by taking over the
training loop. `mldna` doesn't. It bets that most teams won't adopt a tool
that forces a new way of working — so it stays out of the loop entirely and
proves lineage from the outside, by *fingerprinting and gating* instead of
*caging*.

## Philosophy: freedom + guarantees

> A lineage system earns trust not by being unbreakable, but by never
> overstating what it knows.

Two rules, held in tension on purpose:

- **Zero workflow prescription.** Train however you want. Any framework, any
  script structure, model files saved anywhere with any name.
- **Strong guarantees anyway.** Every claim `mldna` makes about a model is
  backed by a cryptographic fact — a content hash, a clean git tree, a
  reachable commit — never by convention or trust.

And critically: when a guarantee *can't* be made at full strength, `mldna`
never hides it. A run trained on a dirty working tree, a model saved after
training ended, a converted file lacking provenance — all of these are
allowed to exist, but they are permanently and visibly labeled as weaker
links, not silently smoothed over.

## What it actually guarantees

1. **Content-addressed resolution.** Every published model resolves to
   exactly one training run via the SHA-256 of the model file itself. Rename
   it, move it, copy it to another machine — the lineage travels with the
   bytes. Anything not captured inside a tracked run, or modified afterward,
   is unpublishable *by construction*, not by policy.
2. **Lineage that's true at the moment it matters.** Training refuses to run
   against a dirty git tree, so the recorded commit is a real fingerprint of
   the code that executed. Publishing can push an existing commit to make it
   reachable — it can never create one. History is recorded, never
   manufactured after the fact.
3. **Honest confidence labels, always.** In-run saves vs. post-run saves,
   dirty runs, conversion provenance — all recorded as tags and surfaced at
   publish time, before a version is minted, not buried in a log somewhere.
4. **Concurrency-safe by default.** A per-directory run lock turns silent
   lineage corruption from concurrent runs into a loud, explanatory error
   instead.
5. **pip-grade discipline on the consuming side.** Consumers pin exact
   model versions in a reviewed lockfile, fetch only the format they need,
   verify checksums, and swap in atomically. Nothing changes in a consuming
   app without a diff someone reviewed.

## How the pieces divide labor

Each system versions exactly one thing. No system does another's job — most
importantly, **model binaries never enter git.**

| Concern | System of record |
|---|---|
| Code | Git |
| Data | DVC, on blob storage |
| Runs, metrics, lineage tags | MLflow (local file store, or Azure ML) |
| Published model versions | A model registry (immutable v1..N) |
| What a consumer actually runs | `models.lock.yaml`, committed in the consumer's repo |

```
consumer app
  └─ models.lock.yaml pins → registry version 9
        └─ registered from → training run a1b2c3
              ├─ tag git_commit      → exact code snapshot (verified clean at train time)
              ├─ tag dvc_data_hash   → exact dataset snapshot
              ├─ tag model_sha256   → exact bytes of the published file
              ├─ tag user / published_by
              └─ captured environment (pip freeze) → exact dependency set
```

Every link above is verified **at the moment it's created** — never
reconstructed after the fact from logs or memory.

### Two-phase lifecycle: runs are cheap, publishes are not

- **Tracking is automatic and immediate.** The moment training starts,
  params, metrics, and lineage tags stream out. Runs are free to be noisy —
  create dozens a day, nobody's grading you on them.
- **Publishing is manual and deliberate.** A run only becomes a consumable,
  registered model version when a developer explicitly runs `mldna publish`,
  which walks the full gate sequence and refuses anything that doesn't hold
  up.

## Install

```bash
pip install mldna-lineage                 # local-only: git + DVC + local MLflow
pip install "mldna-lineage[azure]"        # adds the Azure ML backend
```

The distribution is `mldna-lineage`; the import name and CLI are just `mldna`
(`import mldna`, `mldna --help`) — the same split as `python-dotenv` →
`import dotenv`.

`mldna.core` never imports anything from `azure.*` — the local-only install
has zero Azure dependencies, and the Azure backend is a strict additive
layer on top.

## Quickstart

```bash
# Producer repo (where a model gets trained)
mldna init my-model          # scaffold dataset/, models/, mldna.yaml, .gitignore
mldna push -m "add training data"   # dvc add + git commit + dvc push + git push

# in your training script:
#   with mldna.track_run():
#       ... train, save your model file anywhere under models/ ...

mldna publish models/model.pkl my-model     # runs the full publish gate chain
```

```bash
# Consumer repo (where a published model gets used)
mldna init --consumer
# edit models.lock.yaml: pin { model: my-model, version: 9, format: pkl }
mldna fetch                  # verifies checksums, swaps in atomically
```

## CLI surface

| Command | What it does |
|---|---|
| `mldna init [name] [--consumer] [--azure]` | Scaffold a producer or consumer repo |
| `mldna push` | `dvc add` → commit → `dvc push` → `git push`, idempotently |
| `mldna publish <file> <model_name>` | Resolve the run by content hash, run every gate, register a version |
| `mldna fetch` | Pull every pinned version in `models.lock.yaml`, checksum-verified |
| `mldna pull` | Producer-side: sync code/data, then pull this commit's published bundle if any |
| `mldna runs [--last N]` | Table of recent runs for this project |
| `mldna doctor` | Independent health checks: tools, auth, git remote, `.gitignore`, committed secrets |
| `mldna configure aml-auth` / `configure blob-auth` | Store shared-secret credentials for machines without personal Azure AD access |

Python API: `mldna.track_run()` (context manager around training),
`mldna.track_convert()` (record a converted-format file's provenance against
its source model).

## Governance without a promotion ceremony

There are no `staging` / `production` registry stages. Promotion is just a
pull request against `models.lock.yaml` in the consuming repo — reviewed like
any other code change, with the git history of that file as the full audit
trail of what ran where and who approved it. The one exception is a kill
switch: an admin-set `recalled` tag on a bad version, which consumer tooling
*fails* on (never just warns) if it's ever pinned.

## Project status

Actively under construction. The core local-first loop (`init`, `push`,
`publish`, `fetch`, `runs`, `doctor`) and the Azure ML adapter are both
implemented and tested end-to-end against real infrastructure. See the
commit history for the current phase of work.

## License

MIT — see [LICENSE](LICENSE).
