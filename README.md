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
  script structure, model files saved with any name — anywhere except the
  directories `mldna` treats as disposable scratch and never scans
  (`outputs/`, `mlruns/`, `dataset/`).
- **Strong guarantees anyway.** Every claim `mldna` makes about a model is
  backed by a cryptographic fact — a content hash, a clean git tree, a
  reachable commit — never by convention or trust.

And critically: when a guarantee *can't* be made at full strength, `mldna`
never hides it. A run trained on a dirty working tree, a run whose completion
was recorded after the fact with `mldna attach`, a converted file whose
numerical equivalence was never checked — all of these are allowed to exist,
but they are permanently and visibly labeled as weaker links, not silently
smoothed over. A model saved outside a `with mldna.track_run():` block is not
a weaker link: it is unpublishable.

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
3. **Honest confidence labels, always.** Dirty runs, runs attached after the
   fact, conversion provenance — all recorded as tags and surfaced at
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
              ├─ tag dvc_data_hash__<n> → exact dataset snapshot (one per DVC out)
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
mldna stage                  # dvc add + git add (honors .gitignore), then lists what will be committed
# review the staged files in your editor's source control view; add unwanted ones to .gitignore and re-run stage
mldna push -m "add training data"   # shows the files, confirms, then git commit + dvc push + git push

# in your training script:
#   with mldna.track_run():
#       ... train, save your model file anywhere under models/ ...

mldna publish models/model.pkl              # runs the full publish gate chain
```

```bash
# Consumer repo (where a published model gets used)
mldna init --consumer
# edit models.lock.yaml: pin  <namespace>-my-model: { id: <model-id>, version: 9, format: pkl, dest: ./models/my-model }
mldna fetch                  # verifies checksums, swaps in atomically
```

The name in the lockfile is the registry name: the producer's `project.namespace`
and model name joined by a dash (namespace `acme`, model `my-model` →
`acme-my-model`). With no namespace set it is just the model name.

The scaffolded `mldna.yaml` carries a commented block documenting the optional
settings `mldna` parses but does not require — `tracking.model_globs`,
`tracking.scan_exclude` and `publish.require_formats`. Uncomment what you need;
an untouched scaffold behaves exactly as if they were absent.

`track_run` does not record the command line by default: arguments such as
`--api-key` would become a run tag readable by everyone with access to the
tracking server. Set `tracking.record_argv: true` to opt in.

If your project saves XGBoost or LightGBM models to `.json` or `.txt` (their
native defaults), add these patterns to `tracking.model_globs`. These extensions
are not in the built-in list because they are too generic — every `config.json`
and `requirements.txt` would otherwise be claimed as a model file. In `mldna.yaml`:

```yaml
tracking:
  model_globs:
    - "*.json"      # XGBoost
    - "*.txt"       # LightGBM
```

Config patterns in this field append to the defaults, so you keep the full
built-in set plus your custom globs.

### Local mode: pointing a consumer at the producer's store

With no tracking backend configured, MLflow runs and published bundles live in
`./mlruns` **of the repo that created them**. A consumer repo is a different
directory, so by default `mldna fetch` there would look in its own empty store
and refuse with "not found in the registry". Give the consumer a `filesystem:`
section naming the producer's store:

```yaml
# consumer repo — mldna.yaml
project:
  repo_role: consumer
filesystem:
  path: "${MLDNA_TRACKING_ROOT}/fraud-detector/mlruns"   # or ../fraud-detector/mlruns
```

Any directory both repos can read works — a sibling folder, a shared drive, an
NFS/SMB mount. The path may be absolute, relative to the repo root, or use
environment variables and `~`, so one committed value resolves per machine. It
must already exist: `mldna` refuses with the resolved absolute path rather than
create an empty store (which would make a typo or an unexpanded variable look
like a missing model version). `mldna runs`/`fetch`/`publish` report this as
`Tracking mode: filesystem store` — never as a remote server.

**Local mode is single-user.** Each repository clone keeps its own gitignored
`mlruns/` store, so two developers training in local mode each have a different
"churn-model v1" and a pinned model version resolves to whichever copy exists
in that particular developer's store. A shared team setup needs either a
filesystem store both repositories can read from, or a real `mlflow server`.

MLflow's file-based store has no locking, so a shared folder is safe for the
producer → consumer handoff and read-mostly sharing. If several developers need
to create runs in the same store concurrently, this is a corruption risk — run a
plain `mlflow server` and use the `mlflow:` section instead.

## Model id

Registry names are user-chosen, so two repos can collide on `<namespace>-<model>`.
Each producer project therefore carries an immutable `project.model_id` in
`mldna.yaml`: `mldna init` generates one, and existing projects run `mldna id`
once and commit the change. `publish` refuses without it, stamps it on the
registered model, and refuses if that registry name already belongs to another
id. Consumers add the printed `id:` to their `models.lock.yaml` pin and `fetch`
refuses on a mismatch; a pin without `id:` still works but is flagged as
unverified.

## CLI surface

| Command | What it does |
|---|---|
| `mldna init [name] [--consumer] [--azure]` | Scaffold a producer or consumer repo |
| `mldna stage` | `dvc add` → `git add`, unstages anything now in `.gitignore`, then lists the staged files for review; commits nothing, safe to re-run |
| `mldna push -m <msg>` | Commit what is staged → `dvc push` → `git push`, idempotently; stages nothing itself; lists the files and asks `Commit these N files? [y/N]` every time, flagging large or model-shaped files and a shrinking dataset (`--yes` to skip the question) |
| `mldna id` | Give the project its unique model id in `mldna.yaml` (idempotent; commit the result) |
| `mldna publish <file> [--formats DIR...]` | Resolve the run by content hash, run every gate, register a version (model name comes from `mldna.yaml`) |
| `mldna fetch` | Pull every pinned version in `models.lock.yaml`, checksum-verified |
| `mldna pull` | Producer-side: sync code/data, then pull this commit's published bundle if any |
| `mldna runs [--last N]` | Table of recent runs for this project (the KEY METRIC column stays `-` until you set `tracking.key_metric: <name>` in `mldna.yaml` — it never guesses which metric you meant) |
| `mldna doctor` | Independent health checks: tools, auth, git remote, `.gitignore`, large tracked files |
| `mldna configure aml-auth` / `configure blob-auth` | Store shared-secret credentials for machines without personal Azure AD access |
| `mldna ui [--port N] [--no-browser]` | Read-only local web view of runs, code and data changes, line history and published versions (needs the `[ui]` extra) |

Python API: `mldna.track_run()` (context manager around training),
`mldna.track_convert()` (record a converted-format file's provenance against
its source model).

## Seeing what changed: `mldna ui`

```bash
pip install "mldna-lineage[ui]"
mldna ui                     # opens http://127.0.0.1:8765/
```

A local, read-only page for one producer project:

- **Runs** — the 200 most recent, each with its commit message and badges for
  whether code or data changed since the run before.
- **Code changes** — the commits between two runs (the *why*), then a
  git-style diff. A commit that only moved a `.dvc` pointer is shown as a data
  change, not a code change.
- **Line history** — every line of a file with the commit that last changed it
  and that commit's message; lines changed since the previous run are
  highlighted.
- **Data changes** — files and size per dataset before and after, read from the
  `.dvc` pointers, so the data itself need not be downloaded.
- **Models** — every published version, who published it, and recalled or
  abandoned labels.

It never writes anything: GET routes only, a fixed set of read-only git/dvc
commands, bound to `127.0.0.1`. Commits missing from your clone are reported as
such, never guessed. Details: [src/mldna/ui/README.md](https://github.com/PranavTyagi-3/mldna/blob/master/src/mldna/ui/README.md).

## Governance without a promotion ceremony

There are no `staging` / `production` registry stages. Promotion is just a
pull request against `models.lock.yaml` in the consuming repo — reviewed like
any other code change, with the git history of that file as the full audit
trail of what ran where and who approved it. The one exception is a kill
switch: an admin-set `recalled` tag on a bad version, which consumer tooling
*fails* on (never just warns) if it's ever pinned — including a re-run of
`mldna fetch` when the recalled model is already on disk. A running app is not
stopped; the failed fetch is what must stop the next deploy, so treat a
non-zero exit from `mldna fetch` in CI as a failed deploy.

## Project status

Actively under construction. The core local-first loop (`init`, `push`,
`publish`, `fetch`, `runs`, `doctor`) and the Azure ML adapter are both
implemented and tested end-to-end against real infrastructure. See the
commit history for the current phase of work.

## License

MIT — see [LICENSE](LICENSE).
