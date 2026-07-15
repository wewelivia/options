const $ = (id) => document.getElementById(id);
const API = "";

const PLOT_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "#c7d4ec", family: "-apple-system,Segoe UI,Roboto,sans-serif", size: 12 },
  margin: { l: 54, r: 20, t: 40, b: 46 },
  xaxis: { gridcolor: "#1c2947", zerolinecolor: "#22304f" },
  yaxis: { gridcolor: "#1c2947", zerolinecolor: "#22304f" },
  legend: { orientation: "h", y: -0.22 },
};
const PLOT_CONFIG = { displayModeBar: false, responsive: true };

async function loadPresets() {
  try {
    const r = await fetch(`${API}/api/presets`);
    const d = await r.json();
    const dl = $("preset-list");
    Object.values(d.groups).flat().forEach((name) => {
      const o = document.createElement("option"); o.value = name; dl.appendChild(o);
    });
    window.EXAMPLES = d.example_conditions || {};
  } catch (e) { /* non-fatal */ }
}

async function loadHealth() {
  try {
    const r = await fetch(`${API}/api/health`);
    const d = await r.json();
    const b = $("source-badge");
    if (d.data_source === "bloomberg") { b.textContent = "● Bloomberg live"; b.className = "badge live"; }
    else { b.textContent = "● Synthetic surface (no Terminal)"; b.className = "badge mock"; }
  } catch (e) {
    $("source-badge").textContent = "● backend offline"; $("source-badge").className = "badge mock";
  }
}

async function refreshChain() {
  const und = $("underlying").value.trim();
  if (!und) return;
  try {
    const r = await fetch(`${API}/api/chain?underlying=${encodeURIComponent(und)}`);
    if (!r.ok) return;
    const d = await r.json();
    const sel = $("expiry");
    sel.innerHTML = '<option value="">Nearest to target date</option>';
    d.expiries.forEach((e) => {
      const o = document.createElement("option");
      o.value = e.expiry;
      o.textContent = `${e.expiry}  (T=${e.T.toFixed(2)}y, F=${fmtNum(e.forward)}, ${e.n_strikes} strikes)`;
      sel.appendChild(o);
    });
    // Auto-tick the percent box for rates
    $("force_pct").checked = d.asset_class === "RATES";
    const ex = (window.EXAMPLES || {})[und];
    $("hint").innerHTML = `Asset class <b>${d.asset_class}</b> · ${d.expiries.length} expiries loaded`
      + (ex ? ` · try condition: <b>${ex}</b>` : "");
  } catch (e) { /* ignore */ }
}

function fmtNum(x) {
  if (x == null || isNaN(x)) return "—";
  const a = Math.abs(x);
  if (a >= 1000) return x.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (a >= 10) return x.toFixed(2);
  return x.toFixed(4);
}

async function run() {
  const btn = $("run"); btn.disabled = true; btn.textContent = "Computing…";
  $("error").classList.add("hidden");
  const body = {
    underlying: $("underlying").value.trim(),
    condition: $("condition").value.trim(),
    r: parseFloat($("rate").value || "0"),
    force_percent: $("force_pct").checked,
  };
  const betaV = $("beta").value.trim();
  if (betaV !== "") body.beta = parseFloat(betaV);
  const expV = $("expiry").value;
  if (expV) body.expiry = expV;

  try {
    const r = await fetch(`${API}/api/distribution`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || "Request failed");
    render(d);
  } catch (e) {
    $("error").textContent = "Error: " + e.message;
    $("error").classList.remove("hidden");
  } finally {
    btn.disabled = false; btn.textContent = "Compute probability";
  }
}

function render(d) {
  $("result").classList.remove("hidden");
  const pct = (d.probability * 100).toFixed(1) + "%";
  $("prob").textContent = pct;
  $("prob-cond").textContent = `P( ${d.underlying} ${d.condition} )`;
  $("odds").textContent = `Complement ${(d.complement*100).toFixed(1)}% · implied odds ${d.odds}`;

  $("m-und").textContent = d.underlying;
  $("m-ac").textContent = d.asset_class;
  $("m-exp").textContent = d.expiry;
  $("m-t").textContent = d.T.toFixed(3);
  $("m-fwd").textContent = fmtNum(d.forward);
  $("m-src").textContent = d.source === "bloomberg" ? "Bloomberg" : "Synthetic";

  const unit = d.is_percent ? "%" : "";
  const thrLine = (v, name, color) => ({
    type: "line", x0: v, x1: v, yref: "paper", y0: 0, y1: 1,
    line: { color, width: 1.6, dash: "dash" },
  });
  const shapes = [];
  const annos = [];
  if (d.direction === "between") {
    shapes.push(thrLine(d.threshold, "lo", "#ffb020"), thrLine(d.threshold_hi, "hi", "#ffb020"));
  } else {
    shapes.push(thrLine(d.threshold, "thr", "#ffb020"));
  }
  shapes.push({ type: "line", x0: d.forward, x1: d.forward, yref: "paper", y0: 0, y1: 1,
    line: { color: "#4da3ff", width: 1.2, dash: "dot" } });

  // --- PDF with shaded event region ---
  const g = d.grid, pdf = d.pdf;
  let mask;
  if (d.direction === "above") mask = g.map((x) => x >= d.threshold);
  else if (d.direction === "below") mask = g.map((x) => x <= d.threshold);
  else mask = g.map((x) => x >= d.threshold && x <= d.threshold_hi);
  const fx = [], fy = [];
  g.forEach((x, i) => { if (mask[i]) { fx.push(x); fy.push(pdf[i]); } });

  Plotly.newPlot("chart-dist", [
    { x: g, y: pdf, type: "scatter", mode: "lines", name: "Risk-neutral PDF",
      line: { color: "#4da3ff", width: 2 } },
    { x: fx, y: fy, type: "scatter", mode: "lines", name: "Event region",
      fill: "tozeroy", line: { color: "#38d39f", width: 0 },
      fillcolor: "rgba(56,211,159,.35)" },
  ], { ...PLOT_LAYOUT, title: "Risk-neutral density (PDF)", shapes,
       xaxis: { ...PLOT_LAYOUT.xaxis, title: `Level${unit ? " ("+unit+")" : ""}` },
       yaxis: { ...PLOT_LAYOUT.yaxis, title: "Density" } }, PLOT_CONFIG);

  // --- CDF ---
  Plotly.newPlot("chart-cdf", [
    { x: g, y: d.cdf, type: "scatter", mode: "lines", name: "CDF",
      line: { color: "#38d39f", width: 2 } },
  ], { ...PLOT_LAYOUT, title: "Cumulative distribution (CDF)", shapes,
       xaxis: { ...PLOT_LAYOUT.xaxis, title: `Level${unit ? " ("+unit+")" : ""}` },
       yaxis: { ...PLOT_LAYOUT.yaxis, title: "P(S ≤ x)", range: [0, 1] } }, PLOT_CONFIG);

  // --- Smile: market vs fitted ---
  const sk = d.smile.map((s) => s.strike);
  Plotly.newPlot("chart-smile", [
    { x: sk, y: d.smile.map((s) => s.market_vol * 100), mode: "markers", name: "Market IV",
      marker: { color: "#ffb020", size: 7 } },
    { x: sk, y: d.smile.map((s) => s.fitted_vol * 100), mode: "lines", name: "SABR fit",
      line: { color: "#4da3ff", width: 2 } },
  ], { ...PLOT_LAYOUT, title: "Volatility smile — market vs SABR fit",
       xaxis: { ...PLOT_LAYOUT.xaxis, title: "Strike" },
       yaxis: { ...PLOT_LAYOUT.yaxis, title: "Implied vol (%)" } }, PLOT_CONFIG);

  // --- Stats tables ---
  const s = d.stats;
  const rows = [
    ["Forward", fmtNum(s.forward)], ["Mean", fmtNum(s.mean)],
    ["Mode", fmtNum(s.mode)], ["Median", fmtNum(s.median)],
    ["Std dev", fmtNum(s.std)],
    ["5th pctile", fmtNum(s.p05)], ["25th pctile", fmtNum(s.p25)],
    ["75th pctile", fmtNum(s.p75)], ["95th pctile", fmtNum(s.p95)],
  ];
  $("stats-table").innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
  const sb = d.sabr;
  $("sabr-table").innerHTML = [
    ["α (alpha)", sb.alpha.toFixed(4)], ["β (beta)", sb.beta.toFixed(2)],
    ["ρ (rho)", sb.rho.toFixed(3)], ["ν (nu / vol-of-vol)", sb.nu.toFixed(3)],
    ["Displacement", sb.shift.toFixed(2)], ["Fit RMSE (vol pts)", (sb.rmse*100).toFixed(3)],
  ].map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
}

$("run").addEventListener("click", run);
$("underlying").addEventListener("change", refreshChain);
$("condition").addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });

(async function init() {
  await loadPresets();
  await loadHealth();
  await refreshChain();
})();
