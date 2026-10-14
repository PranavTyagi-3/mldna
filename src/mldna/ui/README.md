# mldna ui

A read-only local web page for one producer project: its runs, what code and
data changed between them, why each line of code is the way it is, and what was
published.

```bash
pip install "mldna-lineage[ui]"
mldna ui                 # opens http://127.0.0.1:8765/ in your browser
mldna ui --port 9000 --no-browser
python -m mldna.ui .     # the same, without the CLI
```

## What it shows

- **Runs**: the 200 most recent runs, newest first: commit and its message,
  whether code or data changed since the run before, key metric, labels
  (dirty, attached after the fact, published versions).
- **Run > Code changes**: the commits between the previous run and this one
  (the "why"), then a git-style diff of every changed file.
- **Run > Line history**: pick a file; every line shows the commit that last
  changed it and that commit's message. Lines changed since the previous run
  are highlighted.
- **Run > Data changes**: per DVC output, files and size before and after,
  read from the `.dvc` pointers (the data itself need not be downloaded); a
  file-level list when dvc can produce one from the local cache.
- **Models**: every registered version, who published it, its formats, the run
  behind it, and recalled / abandoned labels.

Links to the commit or comparison on GitHub, GitLab or Bitbucket appear when
`origin` points there.

## Guarantees

- **Read-only.** GET routes only. Git runs only `log`, `show`, `diff`, `blame`,
  `ls-tree`, `cat-file`, `remote`; dvc runs only `diff`. Every commit hash and
  file path is validated before it reaches a command. Nothing is written to
  git, DVC, MLflow or the registry.
- **Local.** The server binds to `127.0.0.1` only.
- **No secrets.** `.dvc/config.local` is never read; remote URLs are shown
  without credentials.
- **Separate.** Nothing outside `mldna.ui` imports it; without the extra,
  `mldna ui` says how to install it and the rest of mldna is unaffected.

## Layout (for debugging)

| File | Role |
|---|---|
| `gitread.py` | The only place git/dvc are called; allowlist and validation |
| `data.py` | `UiData`: every page's content as plain dicts - call it from a Python prompt |
| `server.py` | FastAPI routes, each a thin wrapper over `UiData`; API docs at `/api/docs` |
| `static/` | `index.html`, `app.js` (plain JS, no build step), `style.css` |

The commits shown are the ones in the clone `mldna ui` runs in; a run whose
commit is missing here says "not in this clone" instead of guessing.
