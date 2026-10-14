// mldna UI - plain JavaScript, no build step. Every value from the server is
// inserted with textContent (via el()), never as HTML.

const app = document.getElementById("app");
let summary = null;

// ---------- helpers ----------

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

async function api(path) {
  const response = await fetch(path);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `${response.status} ${response.statusText}`);
  return body;
}

function show(...nodes) {
  app.replaceChildren(...nodes);
}

function errorBox(error) {
  return el("div", { class: "error" }, error.message || String(error));
}

function short(sha) {
  return sha ? sha.slice(0, 8) : "-";
}

function when(ms) {
  if (!ms) return "-";
  const d = new Date(ms);
  return d.toLocaleDateString() + " " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function size(bytes) {
  if (bytes === null || bytes === undefined) return "-";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let n = bytes, i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${units[i]}`;
}

function commitRef(sha, link) {
  if (!sha) return el("span", { class: "muted" }, "-");
  return link
    ? el("a", { href: link, target: "_blank", rel: "noopener", class: "mono" }, short(sha))
    : el("code", {}, short(sha));
}

function labels(run) {
  return [
    ...run.labels.map((l) => el("span", { class: "badge warn" }, l)),
    ...run.published.map((v) => el("span", { class: "badge ok" }, `published v${v}`)),
  ];
}

// ---------- runs page ----------

async function runsPage() {
  show(el("p", { class: "muted" }, "Loading runs..."));
  let runs;
  try { runs = await api("/api/runs?limit=200"); } catch (e) { return show(errorBox(e)); }
  if (!runs.length) {
    return show(el("h1", {}, "Runs"), el("p", { class: "note" },
      "No runs yet. Train inside `with mldna.track_run():` and they will appear here."));
  }
  const metric = summary.key_metric;
  const rows = runs.map((run) => el("tr", { class: "clickable", onclick: () => go(`#/run/${run.run_id}`) },
    el("td", { class: "mono" }, short(run.run_id)),
    el("td", {}, when(run.started)),
    el("td", { class: "hide-narrow" }, run.user || "-"),
    el("td", {}, commitRef(run.commit, run.commit_link), " ",
      run.subject ? el("span", {}, run.subject) : el("span", { class: "muted" }, run.commit ? "commit not in this clone" : "")),
    el("td", {},
      run.code_changed ? el("span", { class: "badge changed" }, "code") : null,
      run.data_changed ? el("span", { class: "badge changed" }, "data") : null),
    metric ? el("td", { class: "mono" }, run.key_metric ?? "-") : null,
    el("td", {}, labels(run), run.status !== "FINISHED" ? el("span", { class: "badge bad" }, run.status.toLowerCase()) : null),
  ));
  show(
    el("h1", {}, "Runs"),
    el("p", { class: "muted" }, `The ${runs.length} most recent runs. "Changed" compares each run with the one before it.`),
    el("table", {},
      el("thead", {}, el("tr", {},
        el("th", {}, "Run"), el("th", {}, "Started"), el("th", { class: "hide-narrow" }, "User"),
        el("th", {}, "Commit"), el("th", {}, "Changed"), metric ? el("th", {}, metric) : null, el("th", {}, "Labels"))),
      el("tbody", {}, rows)),
  );
}

// ---------- run page ----------

async function runPage(runId, tab) {
  show(el("p", { class: "muted" }, "Loading run..."));
  let run;
  try { run = await api(`/api/runs/${encodeURIComponent(runId)}`); } catch (e) { return show(errorBox(e)); }

  const info = run.commit_info;
  const facts = el("dl", { class: "facts" },
    el("dt", {}, "Started"), el("dd", {}, when(run.started), run.ended ? ` - ended ${when(run.ended)}` : ""),
    el("dt", {}, "User"), el("dd", {}, run.user || "-"),
    el("dt", {}, "Commit"), el("dd", {}, commitRef(run.commit, run.commit_link), " ",
      info ? info.subject : el("span", { class: "muted" }, "not in this clone"),
      run.branch ? el("span", { class: "muted" }, ` on ${run.branch}`) : null),
    el("dt", {}, "Previous run"), el("dd", {},
      run.previous_run ? el("a", { href: `#/run/${run.previous_run}` }, short(run.previous_run)) : el("span", { class: "muted" }, "none - this is the first run")),
    el("dt", {}, "Labels"), el("dd", {}, labels(run).length ? labels(run) : el("span", { class: "muted" }, "none")),
    run.invocation ? [el("dt", {}, "Invocation"), el("dd", { class: "mono" }, run.invocation, el("span", { class: "muted" }, "  (recorded, not verified)"))] : null,
  );

  const base = `#/run/${run.run_id}`;
  const tabs = el("div", { class: "tabs" },
    el("a", { href: base, class: tab === "code" ? "active" : null }, "Code changes"),
    el("a", { href: `${base}/history`, class: tab === "history" ? "active" : null }, "Line history"),
    el("a", { href: `${base}/data`, class: tab === "data" ? "active" : null }, "Data changes"),
    el("a", { href: `${base}/details`, class: tab === "details" ? "active" : null }, "Params, metrics, models"),
  );
  const body = el("div", {}, el("p", { class: "muted" }, "Loading..."));
  show(el("h1", {}, "Run ", el("span", { class: "mono" }, short(run.run_id))), facts, tabs, body);

  const render = { code: codeTab, history: historyTab, data: dataTab, details: detailsTab }[tab];
  try { await render(run, body); } catch (e) { body.replaceChildren(errorBox(e)); }
}

async function codeTab(run, body) {
  if (!run.previous_run) return body.replaceChildren(el("p", { class: "note" }, "This is the first run, so there is nothing to compare its code with."));
  if (!run.commit || !run.previous_commit) return body.replaceChildren(el("p", { class: "note" }, "One of the two runs recorded no commit."));
  if (run.commit === run.previous_commit) {
    return body.replaceChildren(el("p", { class: "note" }, `Same commit as the previous run (${short(run.commit)}) - the code did not change.`));
  }
  const result = await api(`/api/code-diff?old=${run.previous_commit}&new=${run.commit}`);
  const dataNote = result.pointers.length
    ? el("p", { class: "note" }, `DVC pointer file(s) changed too (${result.pointers.join(", ")}) - those are data, shown under `,
        el("a", { href: `#/run/${run.run_id}/data` }, "Data changes"), ".")
    : null;
  if (!run.code_changed) {
    return body.replaceChildren(el("p", { class: "note" }, "The code did not change - only DVC pointer files did, so this is a data change. See ",
      el("a", { href: `#/run/${run.run_id}/data` }, "Data changes"), "."),
      el("h2", {}, "Commits since the previous run"), commitList(result.commits));
  }
  body.replaceChildren(
    el("p", { class: "muted" }, `Changes from the previous run's commit ${short(result.old)} to this run's commit ${short(result.new)}.`,
      result.compare_link ? [" ", el("a", { href: result.compare_link, target: "_blank", rel: "noopener" }, "Open on the remote")] : null),
    el("h2", {}, `Why: ${result.commits.length} commit(s)`),
    commitList(result.commits),
    el("h2", {}, "What changed"),
    dataNote,
    renderDiff(result.diff),
  );
}

function commitList(commits) {
  return el("ul", { class: "commit-list" }, commits.map((c) =>
    el("li", {}, el("code", {}, c.short), " ", c.subject, el("span", { class: "muted" }, ` - ${c.author}, ${new Date(c.date).toLocaleString()}`))));
}

function renderDiff(text) {
  if (!text.trim()) return el("p", { class: "note" }, "No file changes.");
  const files = [];
  let file = null, oldNo = 0, newNo = 0;
  for (const line of text.split("\n")) {
    if (line.startsWith("diff --git ")) {
      const match = line.match(/^diff --git a\/(.*) b\/(.*)$/);
      file = { name: match ? match[2] : line, rows: [], note: "" };
      files.push(file);
    } else if (!file) {
      continue;
    } else if (line.startsWith("@@")) {
      const m = line.match(/^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)$/);
      if (m) { oldNo = +m[1]; newNo = +m[2]; }
      file.rows.push(el("tr", { class: "hunk" }, el("td", { colspan: 3 }, line)));
    } else if (line.startsWith("new file")) {
      file.note = "added";
    } else if (line.startsWith("deleted file")) {
      file.note = "deleted";
    } else if (line.startsWith("Binary files")) {
      file.note = "binary";
    } else if (line.startsWith("+++") || line.startsWith("---") || line.startsWith("index ") ||
               line.startsWith("similarity") || line.startsWith("rename ") || line.startsWith("old mode") ||
               line.startsWith("new mode") || line.startsWith("\\")) {
      continue;
    } else if (line.startsWith("+")) {
      file.rows.push(el("tr", { class: "add" }, el("td", { class: "num" }), el("td", { class: "num" }, newNo++), el("td", {}, line)));
    } else if (line.startsWith("-")) {
      file.rows.push(el("tr", { class: "del" }, el("td", { class: "num" }, oldNo++), el("td", { class: "num" }), el("td", {}, line)));
    } else if (line.startsWith(" ")) {
      file.rows.push(el("tr", {}, el("td", { class: "num" }, oldNo++), el("td", { class: "num" }, newNo++), el("td", {}, line)));
    }
  }
  return el("div", {}, files.map((f) => {
    const added = f.rows.filter((r) => r.className === "add").length;
    const removed = f.rows.filter((r) => r.className === "del").length;
    return el("details", { class: "file", open: f.rows.length < 400 ? "" : null },
      el("summary", {}, f.name, " ",
        el("span", { class: "muted" }, f.note ? `(${f.note}) ` : "", `+${added} -${removed}`)),
      el("table", { class: "diff" }, el("tbody", {}, f.rows)));
  }));
}

async function historyTab(run, body) {
  if (!run.commit || !run.commit_info) {
    return body.replaceChildren(el("p", { class: "note" }, "This run's commit is not in this clone, so line history is not available."));
  }
  const files = await api(`/api/files?commit=${run.commit}`);
  let recent = new Set();
  if (run.previous_commit && run.previous_commit !== run.commit) {
    try {
      const result = await api(`/api/code-diff?old=${run.previous_commit}&new=${run.commit}`);
      recent = new Set(result.commits.map((c) => c.sha));
    } catch (e) { /* history still works without the highlight */ }
  }
  const list = el("div", { class: "file-list" });
  const view = el("div", {}, el("p", { class: "note" }, "Pick a file to see, for every line, the commit that last changed it and why."));
  const filter = el("input", { class: "file-filter", placeholder: "Filter files...", oninput: () => fill(filter.value) });

  function fill(query) {
    const q = query.toLowerCase();
    list.replaceChildren(...files.filter((f) => f.toLowerCase().includes(q)).slice(0, 500).map((f) =>
      el("a", { href: "#", onclick: (event) => { event.preventDefault(); open(f, event.target); } }, f)));
  }

  async function open(path, link) {
    list.querySelectorAll("a.active").forEach((a) => a.classList.remove("active"));
    if (link) link.classList.add("active");
    view.replaceChildren(el("p", { class: "muted" }, `Loading ${path}...`));
    try {
      const result = await api(`/api/blame?commit=${run.commit}&path=${encodeURIComponent(path)}`);
      view.replaceChildren(renderBlame(path, result, recent));
    } catch (e) { view.replaceChildren(errorBox(e)); }
  }

  fill("");
  body.replaceChildren(
    recent.size ? el("p", { class: "muted" }, "Highlighted lines changed since the previous run.") : null,
    el("div", { class: "history-layout" }, el("div", {}, filter, list), view));
}

function renderBlame(path, result, recent) {
  let previous = null;
  const rows = result.lines.map((line, index) => {
    const commit = result.commits[line.commit] || { short: short(line.commit), subject: "" };
    const first = line.commit !== previous;
    previous = line.commit;
    const why = first
      ? el("td", { class: "why", title: `${commit.subject || ""}\n${commit.author || ""}${commit.time ? ", " + new Date(commit.time * 1000).toLocaleString() : ""}` },
          commit.link ? el("a", { href: commit.link, target: "_blank", rel: "noopener" }, commit.short) : el("code", {}, commit.short),
          " ", commit.subject || "")
      : el("td", { class: "why" });
    const classes = [first ? "group-start" : "", recent.has(line.commit) ? "recent" : ""].join(" ").trim();
    return el("tr", { class: classes || null }, why, el("td", { class: "num" }, index + 1), el("td", { class: "code" }, line.text));
  });
  return el("div", {},
    el("h2", { class: "mono" }, path),
    el("table", { class: "blame" }, el("tbody", {}, rows)));
}

async function dataTab(run, body) {
  if (!run.previous_run) return body.replaceChildren(el("p", { class: "note" }, "This is the first run, so there is nothing to compare its data with."));
  const before = JSON.stringify(run.previous_data || {});
  const after = JSON.stringify(run.data || {});
  if (before === after) {
    return body.replaceChildren(el("p", { class: "note" }, "The data is identical to the previous run's (same DVC hashes)."));
  }
  if (!run.commit || !run.previous_commit) return body.replaceChildren(el("p", { class: "note" }, "One of the two runs recorded no commit."));
  body.replaceChildren(el("p", { class: "muted" }, "Comparing the data pointers... (file-level detail asks dvc, which can take a few seconds)"));
  const result = await api(`/api/data-diff?old=${run.previous_commit}&new=${run.commit}`);
  const outs = el("table", {},
    el("thead", {}, el("tr", {}, el("th", {}, "Data"), el("th", {}, "Files"), el("th", {}, "Size"), el("th", {}, "Change"))),
    el("tbody", {}, result.outs.map((o) => {
      const a = o.before || {}, b = o.after || {};
      const change = !o.before ? "added" : !o.after ? "removed" : "changed";
      const shrank = o.before && o.after && (b.size ?? 0) < (a.size ?? 0) / 2;
      return el("tr", {},
        el("td", { class: "mono" }, o.path, el("span", { class: "muted" }, ` (${o.pointer})`)),
        el("td", {}, `${a.nfiles ?? "-"} -> ${b.nfiles ?? "-"}`),
        el("td", {}, `${size(a.size)} -> ${size(b.size)}`),
        el("td", {}, el("span", { class: `badge ${shrank ? "bad" : "changed"}` }, shrank ? "shrank sharply" : change)));
    })));
  const parts = [el("h2", {}, "Datasets"), outs];
  if (result.files) {
    const groups = ["added", "deleted", "modified", "renamed"].filter((k) => (result.files[k] || []).length);
    parts.push(el("h2", {}, "Files"));
    parts.push(groups.length
      ? el("ul", { class: "commit-list" }, groups.flatMap((k) => result.files[k].map((f) =>
          el("li", {}, el("span", { class: "badge changed" }, k), " ",
            el("code", {}, typeof f.path === "string" ? f.path : `${f.path.old} -> ${f.path.new}`)))))
      : el("p", { class: "note" }, "dvc reports no file-level differences."));
  } else {
    parts.push(el("p", { class: "note" }, "File-level detail is not available: dvc could not list the files from this machine's cache. Run `dvc fetch` for both commits to see it."));
  }
  body.replaceChildren(...parts);
}

async function detailsTab(run, body) {
  const table = (obj) => Object.keys(obj).length
    ? el("table", {}, el("tbody", {}, Object.entries(obj).map(([k, v]) => el("tr", {}, el("td", { class: "mono" }, k), el("td", { class: "mono" }, v)))))
    : el("p", { class: "muted" }, "none");
  body.replaceChildren(
    el("h2", {}, "Models saved in this run"),
    run.models.length
      ? el("table", {}, el("tbody", {}, run.models.map((m) => el("tr", {}, el("td", { class: "mono" }, m.path), el("td", { class: "mono muted" }, m.sha256)))))
      : el("p", { class: "muted" }, "none recorded"),
    el("h2", {}, "Metrics"), table(run.metrics),
    el("h2", {}, "Params"), table(run.params),
    el("h2", {}, "Data (DVC hashes)"), table(run.data),
  );
}

// ---------- models page ----------

async function modelsPage() {
  show(el("p", { class: "muted" }, "Loading models..."));
  let result;
  try { result = await api("/api/models"); } catch (e) { return show(errorBox(e)); }
  if (!result.registered) {
    return show(el("h1", {}, result.name), el("p", { class: "note" }, "Nothing published yet. `mldna publish <model file>` registers the first version."));
  }
  show(
    el("h1", {}, result.name),
    el("dl", { class: "facts" },
      el("dt", {}, "Model id"), el("dd", { class: "mono" }, result.model_id || "-"),
      el("dt", {}, "Published from"), el("dd", { class: "mono" }, result.origin || "-")),
    el("table", {},
      el("thead", {}, el("tr", {}, el("th", {}, "Version"), el("th", {}, "Published"), el("th", {}, "By"),
        el("th", {}, "Formats"), el("th", {}, "Run"), el("th", {}, "Labels"))),
      el("tbody", {}, result.versions.map((v) => el("tr", {},
        el("td", {}, `v${v.version}`),
        el("td", {}, when(v.created)),
        el("td", {}, v.published_by || "-"),
        el("td", { class: "mono" }, v.formats || "-"),
        el("td", {}, v.run_id ? el("a", { href: `#/run/${v.run_id}`, class: "mono" }, short(v.run_id)) : "-"),
        el("td", {}, v.labels.map((l) => el("span", { class: "badge bad" }, l))))))));
}

// ---------- routing ----------

function go(hash) {
  location.hash = hash;
}

async function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  document.querySelectorAll("header nav a").forEach((a) =>
    a.classList.toggle("active", a.dataset.nav === (parts[0] === "models" ? "models" : "runs")));
  if (parts[0] === "models") return modelsPage();
  if (parts[0] === "run" && parts[1]) return runPage(parts[1], parts[2] || "code");
  return runsPage();
}

async function start() {
  try {
    summary = await api("/api/summary");
  } catch (e) {
    return show(errorBox(e));
  }
  document.getElementById("model-name").textContent = `${summary.model_name} - ${summary.tracking_mode}`;
  window.addEventListener("hashchange", route);
  route();
}

start();
