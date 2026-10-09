"use strict";

// ---------------------------------------------------------------- state ----

const DEFAULT_TASKS = [
  "sciq:rc::olmo3", "arc_easy:rc::olmes:full", "piqa:rc::olmes:full", "mmlu:rc::olmes",
  "gsm8k::olmes", "triviaqa::olmes", "gsm8k:bpb::olmes", "triviaqa:bpb::olmes",
  "mbpp:3shot:bpb::none",
];

const S = {
  data: null,
  selected: [],        // run ids, in sidebar order
  tab: "val",
  x: "tokens",         // tokens | step
  range: "all",        // all | half | quarter
  unit: "tok",         // tok (nats/token) | char (bits/char)
  mode: "abs",         // abs | delta
  ref: "",             // reference run for deltas
  metric: "score",     // score | bpb (downstream)
  tasks: null,         // array of task aliases, null = defaults
  at: "common",        // table values: common (latest step all runs share, per metric) | final (latest per run) | <step>
};

const $ = (sel) => document.querySelector(sel);
const SVGNS = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids) if (k != null) e.append(k);
  return e;
}

function svg(tag, attrs = {}) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) e.setAttribute(k, v);
  return e;
}

// ---------------------------------------------------------- url state ----

function saveHash() {
  const p = new URLSearchParams();
  p.set("runs", S.selected.join(","));
  for (const k of ["tab", "x", "range", "unit", "mode", "ref", "metric", "at"]) p.set(k, S[k]);
  if (S.tasks) p.set("tasks", S.tasks.join(","));
  history.replaceState(null, "", "#" + p.toString());
}

function loadHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  for (const k of ["tab", "x", "range", "unit", "mode", "ref", "metric", "at"]) if (p.has(k)) S[k] = p.get(k);
  if (p.has("runs")) S.selected = p.get("runs").split(",").filter(Boolean);
  if (p.has("tasks")) S.tasks = p.get("tasks").split(",").filter(Boolean);
}

// ------------------------------------------------------- data helpers ----

const run = (id) => S.data.runs[id];
const color = (id) => `var(--series-${run(id).color || 8})`;
const selectedRuns = () => orderedRunIds().filter((id) => S.selected.includes(id));

function orderedRunIds() {
  return S.data.groups.flatMap((g) => g.runs).filter((id, i, a) => a.indexOf(id) === i);
}

function xOf(r, step) {
  return S.x === "tokens" ? (step * (r.tokens_per_step || 0)) / 1e9 : step;
}

function valPoints(r, dom) {
  const out = [];
  for (const rec of r.val) {
    let v;
    if (dom === "mean") {
      const vals = S.data.val_domains.map((d) => rec.domains[d] && rec.domains[d][S.unit]);
      if (vals.some((x) => x == null)) continue;
      v = vals.reduce((a, b) => a + b, 0) / vals.length;
    } else {
      v = rec.domains[dom] && rec.domains[dom][S.unit];
    }
    if (v != null) out.push([rec.step, v]);
  }
  return out;
}

function isPartial(task, p) {
  return task.expected_subtasks && p[3] != null && p[3] < task.expected_subtasks;
}

function olmesPoints(r, alias, metric) {
  const t = r.olmes[alias];
  if (!t) return [];
  const idx = metric === "bpb" ? 2 : 1;
  return t.points.filter((p) => !isPartial(t, p) && p[idx] != null).map((p) => [p[0], p[idx]]);
}

function taskLowerBetter(alias, metric) {
  if (metric === "bpb") return true;
  for (const id of S.selected) {
    const t = run(id).olmes[alias];
    if (t) return /bits_per_byte|bpb/.test(t.metric || "");
  }
  return false;
}

function trainPoints(r, key) {
  return r.train.step.map((s, i) => [s, r.train[key][i]]).filter((p) => p[1] != null);
}

// Δ vs the reference run at the same step (steps the reference lacks are dropped).
function applyDelta(series, refPts) {
  const ref = new Map(refPts.map((p) => [p[0], p[1]]));
  return series.filter((p) => ref.has(p[0])).map((p) => [p[0], p[1] - ref.get(p[0])]);
}

function seriesFor(getPts, { delta = false } = {}) {
  const refRun = delta && S.ref && S.selected.includes(S.ref) ? run(S.ref) : null;
  const refPts = refRun ? getPts(refRun) : null;
  return selectedRuns().map((id) => {
    const r = run(id);
    let pts = getPts(r);
    if (refPts) pts = applyDelta(pts, refPts);
    return { id, label: r.label, color: color(id), pts: pts.map((p) => [xOf(r, p[0]), p[1], p[0]]) };
  });
}

// ------------------------------------------------------------- format ----

const fmt3 = (v) => (v == null || !isFinite(v) ? "–" : v.toFixed(3));
const fmtSigned = (v) => (v == null ? "–" : (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(v).toFixed(3));
const fmtInt = (v) => Math.round(v).toLocaleString("en-US");
function fmtX(v) {
  if (S.x === "tokens") return (Number.isInteger(+v.toFixed(6)) || Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(1)) + "B";
  return v >= 1000 ? (v / 1000).toFixed(v % 1000 ? 1 : 0) + "k" : String(v);
}
const xTitle = () => (S.x === "tokens" ? "tokens" : "step");

function niceTicks(lo, hi, count = 5) {
  if (!(hi > lo)) { const d = Math.abs(lo) * 0.05 || 1; lo -= d; hi += d; }
  const raw = (hi - lo) / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(+v.toFixed(12));
  return { ticks, step };
}

// enough decimals to tell adjacent ticks apart (0.25 -> 2, 0.5 -> 1, 20 -> 0)
function tickFmt(step) {
  let dec = 0;
  while (dec < 6 && Math.abs(step * 10 ** dec - Math.round(step * 10 ** dec)) > 1e-6) dec++;
  return (v) => v.toFixed(dec);
}

// -------------------------------------------------------------- chart ----

const tooltip = () => $("#tooltip");

function hideTooltip() { tooltip().hidden = true; }

function showTooltip(evt, head, rows) {
  const tt = tooltip();
  tt.replaceChildren(el("div", { class: "tt-head", text: head }));
  for (const r of rows) {
    const key = el("span", { class: "key" }); key.style.background = r.color;
    tt.append(el("div", { class: "tt-row" }, key,
      el("span", { class: "tt-val", text: r.value }), el("span", { class: "tt-name", text: r.label })));
  }
  tt.hidden = false;
  const pad = 14, w = tt.offsetWidth, h = tt.offsetHeight;
  let left = evt.clientX + pad, top = evt.clientY + pad;
  if (left + w > window.innerWidth - 8) left = evt.clientX - w - pad;
  if (top + h > window.innerHeight - 8) top = Math.max(8, window.innerHeight - h - 8);
  tt.style.left = left + "px"; tt.style.top = top + "px";
}

// One card with a line chart. series: [{id,label,color,pts:[[x,y,step]]}]
let clipId = 0;
function lineChart(parent, { title, sub, series, domainX, lowerBetter, zeroLine, fmtVal = fmt3, yIgnoreFirst = 0 }) {
  const card = el("div", { class: "card" }, el("h3", { text: title }), sub ? el("p", { class: "sub", text: sub }) : null);
  parent.append(card);
  const [x0, x1] = domainX;
  const vis = series.map((s) => ({ ...s, pts: s.pts.filter((p) => p[0] >= x0 - 1e-9 && p[0] <= x1 + 1e-9) }));
  // yIgnoreFirst: fraction of the x-range left out of the y-extent (e.g. the loss spike
  // at init); lines there are clipped to the plot area
  const yFrom = x0 + yIgnoreFirst * (x1 - x0);
  const ys = vis.flatMap((s) => s.pts.filter((p) => p[0] >= yFrom).map((p) => p[1]));
  if (zeroLine && ys.length) ys.push(0);
  if (!ys.length) {
    card.append(el("div", { class: "empty", text: "No data for the selected runs in this range." }));
    return;
  }

  const W = Math.max(280, card.clientWidth - 24), H = 230;
  const m = { l: 52, r: 14, t: 10, b: 30 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  let ylo = Math.min(...ys), yhi = Math.max(...ys);
  const padY = (yhi - ylo) * 0.08 || Math.abs(yhi) * 0.02 || 0.01;
  ylo -= padY; yhi += padY;
  const yt = niceTicks(ylo, yhi, 5), xt = niceTicks(x0, x1, Math.max(3, Math.round(pw / 90)));
  ylo = Math.min(ylo, yt.ticks[0]); yhi = Math.max(yhi, yt.ticks[yt.ticks.length - 1]);
  const sx = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * pw;
  const sy = (v) => m.t + (1 - (v - ylo) / (yhi - ylo || 1)) * ph;

  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, height: H, tabindex: 0, role: "img",
    "aria-label": `${title}: line chart; use the Table tab for exact values` });
  const yf = tickFmt(yt.step);
  for (const t of yt.ticks) {
    if (t < ylo - 1e-12 || t > yhi + 1e-12) continue;
    root.append(svg("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t) }));
    const lab = svg("text", { x: m.l - 6, y: sy(t) + 4, "text-anchor": "end" }); lab.textContent = yf(t);
    root.append(lab);
  }
  if (zeroLine) root.append(svg("line", { class: "axisline", x1: m.l, x2: W - m.r, y1: sy(0), y2: sy(0) }));
  root.append(svg("line", { class: "axisline", x1: m.l, x2: W - m.r, y1: m.t + ph, y2: m.t + ph }));
  for (const t of xt.ticks) {
    if (t < x0 - 1e-9 || t > x1 + 1e-9) continue;
    const lab = svg("text", { x: sx(t), y: H - 10, "text-anchor": "middle" }); lab.textContent = fmtX(t);
    root.append(lab);
  }

  const cid = `clip${++clipId}`;
  const defs = svg("defs"), cp = svg("clipPath", { id: cid });
  cp.append(svg("rect", { x: m.l - 6, y: m.t - 6, width: pw + 12, height: ph + 12 }));
  defs.append(cp);
  const plot = svg("g", { "clip-path": `url(#${cid})` });
  root.append(defs, plot);
  for (const s of vis) {
    if (s.pts.length > 1) {
      const d = s.pts.map((p, i) => (i ? "L" : "M") + sx(p[0]).toFixed(1) + "," + sy(p[1]).toFixed(1)).join("");
      plot.append(svg("path", { class: "series", d, stroke: s.color }));
    }
    if (s.pts.length <= 40) {
      for (const p of s.pts) plot.append(svg("circle", { class: "dot", cx: sx(p[0]), cy: sy(p[1]), r: 4, fill: s.color }));
    }
  }

  // hover / focus layer: crosshair snaps to the nearest data x; tooltip lists every run there
  const xs = [...new Set(vis.flatMap((s) => s.pts.map((p) => p[0])))].sort((a, b) => a - b);
  const tol = (x1 - x0) * 0.012;
  const hair = svg("line", { class: "crosshair", y1: m.t, y2: m.t + ph, visibility: "hidden" });
  const marks = svg("g");
  const overlay = svg("rect", { x: m.l, y: m.t, width: pw, height: ph, fill: "transparent" });
  root.append(hair, marks, overlay);
  let cur = -1;

  function at(i, evt) {
    if (i < 0 || i >= xs.length) return;
    cur = i;
    const x = xs[i];
    hair.setAttribute("x1", sx(x)); hair.setAttribute("x2", sx(x)); hair.setAttribute("visibility", "visible");
    marks.replaceChildren();
    const rows = [];
    let step = null;
    for (const s of vis) {
      let best = null;
      for (const p of s.pts) if (Math.abs(p[0] - x) <= tol && (!best || Math.abs(p[0] - x) < Math.abs(best[0] - x))) best = p;
      if (best) {
        step = step ?? best[2];
        marks.append(svg("circle", { class: "dot", cx: sx(best[0]), cy: sy(best[1]), r: 5, fill: s.color }));
      }
      rows.push({ label: s.label, color: s.color, value: best ? fmtVal(best[1]) : "–", raw: best ? best[1] : null });
    }
    rows.sort((a, b) => (a.raw == null) - (b.raw == null) || (lowerBetter ? a.raw - b.raw : b.raw - a.raw));
    const head = S.x === "tokens" ? `${fmtX(x)} tokens · step ${fmtInt(step ?? 0)}` : `step ${fmtInt(x)}`;
    const r = root.getBoundingClientRect();
    const e = evt || { clientX: r.left + (sx(x) / W) * r.width, clientY: r.top + 20 };
    showTooltip(e, head + (lowerBetter ? "  (lower is better)" : "  (higher is better)"), rows);
  }
  function nearest(px) {
    const x = x0 + ((px - m.l) / pw) * (x1 - x0);
    let bi = -1, bd = Infinity;
    xs.forEach((v, i) => { const d = Math.abs(v - x); if (d < bd) { bd = d; bi = i; } });
    return bi;
  }
  function leave() { hair.setAttribute("visibility", "hidden"); marks.replaceChildren(); hideTooltip(); }
  overlay.addEventListener("pointermove", (evt) => {
    const r = root.getBoundingClientRect();
    at(nearest(((evt.clientX - r.left) / r.width) * W), evt);
  });
  overlay.addEventListener("pointerleave", leave);
  root.addEventListener("focus", () => at(cur >= 0 ? cur : xs.length - 1));
  root.addEventListener("blur", leave);
  root.addEventListener("keydown", (evt) => {
    if (evt.key === "ArrowLeft") { at(Math.max(0, cur - 1)); evt.preventDefault(); }
    if (evt.key === "ArrowRight") { at(Math.min(xs.length - 1, cur + 1)); evt.preventDefault(); }
  });
  card.append(root);
}

function xDomain(seriesLists) {
  let max = 0;
  for (const list of seriesLists) for (const s of list) for (const p of s.pts) max = Math.max(max, p[0]);
  const lo = S.range === "half" ? max * 0.5 : S.range === "quarter" ? max * 0.75 : 0;
  return [lo, max || 1];
}

// ------------------------------------------------------------ filters ----

function seg(label, key, options, onChange) {
  const wrap = el("span", { class: "seg", role: "group", "aria-label": label });
  for (const [val, text] of options) {
    wrap.append(el("button", {
      type: "button", "aria-pressed": String(S[key] === val), text,
      onclick: () => { S[key] = val; (onChange || render)(); },
    }));
  }
  return el("label", { class: "f" }, label, wrap);
}

function refSelect() {
  const sel = el("select", { onchange: (e) => { S.ref = e.target.value; render(); } });
  for (const id of selectedRuns()) {
    const o = el("option", { value: id, text: run(id).label });
    if (id === S.ref) o.selected = true;
    sel.append(o);
  }
  return el("label", { class: "f" }, "Reference", sel);
}

function renderFilters() {
  const f = $("#filters");
  f.replaceChildren();
  if (S.tab !== "table") {
    f.append(seg("X axis", "x", [["tokens", "Tokens"], ["step", "Step"]]));
    f.append(seg("Range", "range", [["all", "All"], ["half", "Last 50%"], ["quarter", "Last 25%"]]));
  }
  if (S.tab === "val" || S.tab === "table") f.append(seg("Val metric", "unit", [["tok", "NLL / token"], ["char", "bits / char"]]));
  if (S.tab === "downstream") f.append(seg("Metric", "metric", [["score", "Score"], ["bpb", "Gold BPB"]]));
  if (S.tab === "val" || S.tab === "downstream") f.append(seg("Show", "mode", [["abs", "Absolute"], ["delta", "Δ vs reference"]]));
  if (S.tab === "table") {
    const steps = [...new Set(selectedRuns().flatMap((id) => run(id).val.map((v) => v.step)))].sort((a, b) => a - b);
    const sel = el("select", { onchange: (e) => { S.at = e.target.value; render(); } });
    sel.append(el("option", { value: "common", text: "Latest common step" }));
    sel.append(el("option", { value: "final", text: "Latest per run" }));
    for (const s of steps) sel.append(el("option", { value: String(s), text: `Step ${fmtInt(s)}` }));
    sel.value = ["common", "final", ...steps.map(String)].includes(S.at) ? S.at : "common";
    f.append(el("label", { class: "f" }, "Values at", sel));
  }
  if (S.mode === "delta" || S.tab === "table") f.append(refSelect());
}

// ------------------------------------------------------------- views ----

function viewVal(view) {
  const doms = [...S.data.val_domains, "mean"];
  const delta = S.mode === "delta";
  const lists = doms.map((d) => seriesFor((r) => valPoints(r, d), { delta }));
  const dx = xDomain(lists);
  const unit = S.unit === "tok" ? "NLL per token (nats)" : "bits per char";
  $("#note").textContent = `Held-out Dolmino val sets (1000 docs per source). ${delta ? `Δ = run − ${run(S.ref).label} at the same step; below 0 is better.` : "Lower is better."}`;
  const grid = el("div", { class: "grid" });
  view.append(grid);
  doms.forEach((d, i) => lineChart(grid, {
    title: d === "mean" ? "Mean over sources" : d, sub: delta ? `Δ ${unit}` : unit,
    series: lists[i], domainX: dx, lowerBetter: true, zeroLine: delta,
    fmtVal: delta ? fmtSigned : fmt3,
  }));
}

function availableTasks() {
  const have = new Set(selectedRuns().flatMap((id) => Object.keys(run(id).olmes)));
  return S.data.tasks.filter((t) => have.has(t));
}

function shownTasks() {
  const avail = availableTasks();
  if (S.tasks) return S.tasks.filter((t) => avail.includes(t));
  const def = DEFAULT_TASKS.filter((t) => avail.includes(t));
  return def.length ? def : avail;
}

function viewDownstream(view) {
  const avail = availableTasks();
  const shown = shownTasks();
  const chips = el("div", { class: "chips", role: "group", "aria-label": "Tasks" });
  for (const t of avail) {
    chips.append(el("button", {
      type: "button", class: "chip", "aria-pressed": String(shown.includes(t)), text: t,
      onclick: () => {
        const cur = new Set(shownTasks());
        cur.has(t) ? cur.delete(t) : cur.add(t);
        S.tasks = avail.filter((x) => cur.has(x));
        render();
      },
    }));
  }
  view.append(chips);
  const delta = S.mode === "delta";
  const metric = S.metric;
  const tasks = shown.filter((t) => selectedRuns().some((id) => olmesPoints(run(id), t, metric).length));
  const skipped = shown.filter((t) => !tasks.includes(t));
  $("#note").textContent =
    (metric === "bpb" ? "Gold-answer bits per byte (lower is better). " : "OLMES primary score; BPB tasks are lower-is-better. ") +
    (delta ? `Δ = run − ${run(S.ref).label}. ` : "") +
    "Single dots are one-off evaluations of a final checkpoint." +
    (skipped.length ? ` No ${metric === "bpb" ? "BPB" : "score"} for: ${skipped.join(", ")}.` : "");
  const lists = tasks.map((t) => seriesFor((r) => olmesPoints(r, t, metric), { delta }));
  const dx = xDomain(lists);
  const grid = el("div", { class: "grid" });
  view.append(grid);
  tasks.forEach((t, i) => {
    const lb = taskLowerBetter(t, metric);
    const anyRun = selectedRuns().map((id) => run(id).olmes[t]).find(Boolean);
    const sub = (metric === "bpb" ? "bits_per_byte_corr" : anyRun.metric) + (delta ? " (Δ)" : "");
    lineChart(grid, { title: t, sub, series: lists[i], domainX: dx, lowerBetter: lb, zeroLine: delta, fmtVal: delta ? fmtSigned : fmt3 });
  });
}

function viewTrain(view) {
  const loss = seriesFor((r) => trainPoints(r, "loss"));
  const lr = seriesFor((r) => trainPoints(r, "lr"));
  const dx = xDomain([loss]);
  $("#note").textContent = "Training loss is on each run's own training data, so it is not comparable across different data mixes.";
  const grid = el("div", { class: "grid" });
  view.append(grid);
  lineChart(grid, { title: "Training loss", sub: "mean over 50-step windows; y-range excludes the first 5% (init spike)",
    series: loss, domainX: dx, lowerBetter: true, yIgnoreFirst: S.range === "all" ? 0.05 : 0 });
  lineChart(grid, { title: "Learning rate", sub: "", series: lr, domainX: dx, lowerBetter: false, fmtVal: (v) => v.toExponential(2) });
}

// values of one metric for each run: at an exact step, at the latest step every run
// has (common), or each run's own latest point (final)
function pickRow(ptsPerRun) {
  if (S.at === "final") return ptsPerRun.map((pts) => (pts.length ? pts[pts.length - 1] : null));
  let step = +S.at;
  if (S.at === "common") {
    const sets = ptsPerRun.filter((p) => p.length).map((pts) => new Set(pts.map((p) => p[0])));
    const shared = sets.length ? [...sets[0]].filter((s) => sets.every((st) => st.has(s))) : [];
    if (!shared.length) return ptsPerRun.map(() => null);
    step = Math.max(...shared);
  }
  return ptsPerRun.map((pts) => pts.find((p) => p[0] === step) || null);
}

function viewTable(view) {
  const ids = selectedRuns();
  const unitName = S.unit === "tok" ? "NLL / token" : "bits / char";
  const rows = [];
  rows.push({ section: `Validation ${unitName} (lower is better)` });
  for (const d of [...S.data.val_domains, "mean"]) {
    rows.push({ name: d === "mean" ? "mean over sources" : d, lower: true, get: (r) => valPoints(r, d) });
  }
  const tasks = availableTasks();
  const scoreTasks = tasks.filter((t) => !taskLowerBetter(t, "score"));
  if (scoreTasks.length) {
    rows.push({ section: "Downstream score (higher is better)" });
    for (const t of scoreTasks) rows.push({ name: t, lower: false, get: (r) => olmesPoints(r, t, "score") });
  }
  const bpbTasks = tasks.filter((t) => ids.some((id) => olmesPoints(run(id), t, "bpb").length));
  if (bpbTasks.length) {
    rows.push({ section: "Downstream gold BPB (lower is better)" });
    for (const t of bpbTasks) rows.push({ name: t, lower: true, get: (r) => olmesPoints(r, t, "bpb") });
  }
  $("#note").textContent =
    (S.at === "common" ? "Per metric, the latest step evaluated for every selected run (step shown in the first column). " :
     S.at === "final" ? "Each run's latest evaluated checkpoint; steps can differ between runs (shown under the value). " :
     `Step ${fmtInt(+S.at)} only. `) +
    "Best value per row in bold; Δ is against the reference run.";

  const thead = el("tr", {}, el("th", { text: "metric" }));
  for (const id of ids) {
    const key = el("span", { class: "key" }); key.style.background = color(id);
    thead.append(el("th", {}, key, run(id).label));
  }
  const tbody = el("tbody");
  for (const row of rows) {
    if (row.section) { tbody.append(el("tr", { class: "section" }, el("td", { colspan: ids.length + 1, text: row.section }))); continue; }
    const vals = pickRow(ids.map((id) => row.get(run(id))));
    const nums = vals.filter(Boolean).map((p) => p[1]);
    if (!nums.length) continue;
    const best = row.lower ? Math.min(...nums) : Math.max(...nums);
    const refV = S.ref && ids.includes(S.ref) ? vals[ids.indexOf(S.ref)] : null;
    const steps = new Set(vals.filter(Boolean).map((p) => p[0]));
    const nameCell = el("td", { text: row.name });
    if (S.at === "common") nameCell.append(el("span", { class: "step", text: `step ${fmtInt(vals.find(Boolean)[0])}` }));
    const tr = el("tr", {}, nameCell);
    vals.forEach((p, i) => {
      if (!p) { tr.append(el("td", { class: "num muted", text: "–" })); return; }
      const isBest = nums.length > 1 && p[1] === best;
      const td = el("td", { class: "num" + (isBest ? " best" : ""), title: isBest ? "best in row" : null, text: fmt3(p[1]) });
      if (refV && ids[i] !== S.ref) {
        const d = p[1] - refV[1];
        const better = row.lower ? d < 0 : d > 0;
        const cls = Math.abs(d) < 5e-4 ? "zero" : better ? "good" : "bad";
        const arrow = Math.abs(d) < 5e-4 ? "" : d > 0 ? "▲ " : "▼ ";
        td.append(el("span", { class: "delta " + cls, text: arrow + fmtSigned(d) }));
      }
      if (S.at === "final" && steps.size > 1) td.append(el("span", { class: "step", text: `step ${fmtInt(p[0])}` }));
      tr.append(td);
    });
    tbody.append(tr);
  }
  view.append(el("div", { class: "table-wrap" }, el("table", { class: "results" }, el("thead", {}, thead), tbody)));
}

// ------------------------------------------------------------ sidebar ----

function renderSidebar() {
  const side = $("#sidebar");
  side.replaceChildren();
  for (const g of S.data.groups) {
    const allOn = g.runs.every((id) => S.selected.includes(id));
    const head = el("div", { class: "group-head" }, el("h2", { text: g.title }),
      el("button", { type: "button", text: allOn ? "none" : "all", onclick: () => {
        S.selected = allOn ? S.selected.filter((id) => !g.runs.includes(id)) : [...new Set([...S.selected, ...g.runs])];
        render();
      } }));
    const box = el("div", { class: "group" }, head, g.description ? el("p", { text: g.description }) : null);
    for (const id of g.runs) {
      const r = run(id);
      const key = el("span", { class: "key" }); key.style.background = color(id);
      const lastVal = r.val.length ? r.val[r.val.length - 1].step : null;
      const sub = !r.exists ? "directory not found" :
        `step ${fmtInt(r.max_step)}` + (lastVal ? ` · last val ${fmtInt(lastVal)}` : " · no val yet");
      const cb = el("input", { type: "checkbox", checked: S.selected.includes(id), onchange: (e) => {
        S.selected = e.target.checked ? [...S.selected, id] : S.selected.filter((x) => x !== id);
        render();
      } });
      box.append(el("label", { class: "run" }, cb, key,
        el("span", { class: "label" }, r.label, el("br"), el("span", { class: "sub", text: sub }))));
    }
    side.append(box);
  }
}

function renderLegend() {
  const lg = $("#legend");
  lg.replaceChildren();
  if (S.tab === "table") return;
  for (const id of selectedRuns()) {
    const key = el("span", { class: "key" }); key.style.background = color(id);
    lg.append(el("span", { class: "item" }, key, run(id).label + (S.mode === "delta" && id === S.ref && S.tab !== "train" ? " (reference)" : "")));
  }
}

// ------------------------------------------------------------- render ----

function render() {
  if (!S.data) return;
  const known = orderedRunIds();
  S.selected = S.selected.filter((id) => known.includes(id));
  if (!S.selected.includes(S.ref)) S.ref = selectedRuns()[0] || "";
  if (S.tab === "train" && S.mode === "delta") S.mode = "abs";
  for (const b of document.querySelectorAll(".tabs button")) b.setAttribute("aria-selected", String(b.dataset.tab === S.tab));
  renderSidebar();
  renderFilters();
  renderLegend();
  hideTooltip();
  const view = $("#view");
  view.replaceChildren();
  $("#note").textContent = "";
  if (!S.selected.length) {
    view.append(el("div", { class: "empty", text: "Select runs on the left." }));
  } else {
    ({ val: viewVal, downstream: viewDownstream, train: viewTrain, table: viewTable })[S.tab](view);
  }
  saveHash();
}

async function load(refresh) {
  const main = $("#main");
  main.classList.add("loading");
  try {
    if (window.RESULTS_DATA && !refresh) {
      S.data = window.RESULTS_DATA;
    } else {
      const res = await fetch("api/data" + (refresh ? "?refresh=1" : ""), { cache: "no-store" });
      const d = await res.json();
      if (d.error) throw new Error(d.error);
      S.data = d;
    }
    $("#stamp").textContent = `data collected ${S.data.generated_at}`;
    if (!S.selected.length) S.selected = [...S.data.groups[0].runs];
    render();
  } catch (e) {
    $("#view").replaceChildren(el("div", { class: "empty", text: "Could not load results: " + e.message }));
  } finally {
    main.classList.remove("loading");
  }
}

// --------------------------------------------------------------- init ----

function applyTheme(t) {
  if (t === "auto") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", t);
  $("#theme").textContent = "Theme: " + t;
  localStorage.setItem("rv-theme", t);
}

window.addEventListener("DOMContentLoaded", () => {
  loadHash();
  for (const b of document.querySelectorAll(".tabs button")) b.addEventListener("click", () => { S.tab = b.dataset.tab; render(); });
  $("#refresh").addEventListener("click", () => load(true));
  if (window.RESULTS_DATA) $("#refresh").hidden = true;
  const themes = ["auto", "light", "dark"];
  applyTheme(localStorage.getItem("rv-theme") || "auto");
  $("#theme").addEventListener("click", () => {
    const cur = localStorage.getItem("rv-theme") || "auto";
    applyTheme(themes[(themes.indexOf(cur) + 1) % themes.length]);
  });
  let t = null;
  window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(render, 150); });
  load(false);
});
