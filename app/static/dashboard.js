function el(id){ return document.getElementById(id); }

const chartStore = {};

function fmtNum(value, digits = 2){
  if(value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function fmtInt(value){
  if(value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return String(parseInt(value, 10));
}

function safeArray(value){
  return Array.isArray(value) ? value : [];
}

function card(k, v){
  const d = document.createElement("div");
  d.className = "card";
  d.innerHTML = `<div class="k">${k}</div><div class="v">${v}</div>`;
  return d;
}

function tableFrom(rows, cols){
  let html = "<tr>" + cols.map(c => `<th>${c}</th>`).join("") + "</tr>";
  for(const r of rows){
    html += "<tr>" + cols.map(c => `<td>${r[c] ?? "-"}</td>`).join("") + "</tr>";
  }
  return html;
}

async function rerunAnalysis(){
  const city = window.DASH_CITY;
  const cc = window.DASH_COUNTRY || "";
  const start = window.DASH_START;
  const end = window.DASH_END;

  el("statusBox").textContent = "Re-running analysis...";
  const url =
    `/analyse/${encodeURIComponent(city)}` +
    `?country_code=${encodeURIComponent(cc)}` +
    `&start=${encodeURIComponent(start)}` +
    `&end=${encodeURIComponent(end)}`;

  const res = await fetch(url, { method: "POST" });
  const j = await res.json();

  if(!res.ok){
    el("statusBox").textContent = "Analysis failed: " + (j.error || "unknown");
    return;
  }

  await loadDashboard();
}

function destroyChartIfExists(canvasId){
  if(chartStore[canvasId]){
    chartStore[canvasId].destroy();
    delete chartStore[canvasId];
  }
}

function plotLine(canvasId, labels, values, title){
  destroyChartIfExists(canvasId);
  const ctx = el(canvasId);
  if(!ctx) return;

  chartStore[canvasId] = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{ label: title, data: values, tension: 0.25 }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      resizeDelay: 150,
      plugins: {
        legend: { display: true }
      }
    }
  });
}

function plotBar(canvasId, labels, values, title){
  destroyChartIfExists(canvasId);
  const ctx = el(canvasId);
  if(!ctx) return;

  chartStore[canvasId] = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{ label: title, data: values }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      resizeDelay: 150,
      plugins: {
        legend: { display: true }
      }
    }
  });
}

function renderSummaryCards(payload){
  const cards = el("cards");
  cards.innerHTML = "";

  const s = payload.summary || {};
  const patterns = payload.patterns || {};
  const pSummary = patterns.summary || {};

  const hm = s.hottest_month
    ? `${s.hottest_month.month_name} (${fmtNum(s.hottest_month.tavg_mean, 1)}°C)`
    : "-";

  const cm = s.coolest_month
    ? `${s.coolest_month.month_name} (${fmtNum(s.coolest_month.tavg_mean, 1)}°C)`
    : "-";

  cards.appendChild(card("Annual trend (°C/year)", fmtNum(s.annual_trend_c_per_year ?? 0, 3)));
  cards.appendChild(card("Anomaly days (|z| ≥ 2)", fmtInt(s.anomaly_days_count_abs_ge_2 ?? 0)));
  cards.appendChild(card("Hottest month", hm));
  cards.appendChild(card("Coolest month", cm));

  cards.appendChild(card("Pattern method", patterns.method || "-"));
  cards.appendChild(card("Patterns discovered", fmtInt(s.n_patterns ?? pSummary.n_triclusters ?? 0)));

  if(patterns.engine && patterns.engine.n_supported_windows !== undefined){
    cards.appendChild(card("Supported windows", fmtInt(patterns.engine.n_supported_windows)));
  } else if(pSummary.n_supported_windows !== undefined){
    cards.appendChild(card("Supported windows", fmtInt(pSummary.n_supported_windows)));
  }

  if(payload.data_health){
    const cov = payload.data_health.coverage_ratio;
    cards.appendChild(card("Data coverage", cov !== undefined ? `${fmtNum(cov * 100, 1)}%` : "-"));
  }
}

function renderMonthlyCharts(payload){
  const mb = safeArray(payload.monthly_baseline);
  const labels = mb.map(x => x.month_name);

  plotLine(
    "chartMean",
    labels,
    mb.map(x => Number(x.tavg_mean ?? 0)),
    "Monthly mean tavg (°C)"
  );

  plotBar(
    "chartStd",
    labels,
    mb.map(x => Number(x.tavg_std ?? 0)),
    "Monthly std tavg (°C)"
  );

  plotLine(
    "chartDiurnal",
    labels,
    mb.map(x => Number(x.diurnal_mean ?? 0)),
    "Monthly diurnal mean (°C)"
  );
}

function clusterMetaLine(c){
  const bits = [];

  if(c.type) bits.push(`Type: ${c.type}`);
  if(c.support !== undefined) bits.push(`Support: ${fmtInt(c.support)}`);
  if(c.hvar3 !== undefined) bits.push(`HVar³: ${fmtNum(c.hvar3, 3)}`);

  return bits.join(" | ");
}

function signatureLine(c){
  const sig = safeArray(c.signature_top5);
  if(sig.length === 0) return "-";

  return sig.map(x => {
    const feature = x.feature ?? "-";
    const direction = x.direction ?? x.symbol ?? "-";
    const count = x.count !== undefined ? ` (${x.count})` : "";
    return `${feature}: ${direction}${count}`;
  }).join(" | ");
}

function renderPatterns(payload){
  const clusters = el("clusters");
  clusters.innerHTML = "";

  const patterns = payload.patterns || {};
  const cls = safeArray(patterns.clusters);

  if(cls.length === 0){
    clusters.textContent = "No patterns found.";
    return;
  }

  for(const c of cls){
    const div = document.createElement("div");
    div.className = "cluster";

    const months = safeArray(c.months_names).join(", ");
    const yearsArr = safeArray(c.years);
    const years = yearsArr.slice(0, 10).join(", ") + (yearsArr.length > 10 ? "..." : "");
    const meta = clusterMetaLine(c);
    const sig = signatureLine(c);

    div.innerHTML = `
      <div><span class="badge">${c.label || "Pattern"}</span></div>
      <div style="margin-top:8px;opacity:0.95">${meta || "-"}</div>
      <div style="margin-top:6px;opacity:0.9">Months: ${months || "-"}</div>
      <div style="margin-top:6px;opacity:0.9">Years: ${years || "-"}</div>
      <div style="margin-top:6px;opacity:0.9">Signature: ${sig}</div>
    `;

    clusters.appendChild(div);
  }
}

function renderExtremeTables(payload){
  const ex = payload.extremes || {};

  el("tblAnom").innerHTML = tableFrom(
    safeArray(ex.top_anomaly_days).map(x => ({
      date: x.date,
      anomaly_z: fmtNum(x.anomaly_z, 2),
    })),
    ["date", "anomaly_z"]
  );

  el("tblHot").innerHTML = tableFrom(
    safeArray(ex.top_hot_days).map(x => ({
      date: x.date,
      tavg: fmtNum(x.tavg, 1),
    })),
    ["date", "tavg"]
  );

  el("tblCold").innerHTML = tableFrom(
    safeArray(ex.top_cold_days).map(x => ({
      date: x.date,
      tavg: fmtNum(x.tavg, 1),
    })),
    ["date", "tavg"]
  );
}

async function loadDashboard(){
  const city = window.DASH_CITY;
  const cc = window.DASH_COUNTRY || "";

  el("title").textContent = `Weather Insights — ${city}${cc ? " (" + cc + ")" : ""}`;
  el("subtitle").textContent = `Range: ${window.DASH_START} → ${window.DASH_END}`;

  const res = await fetch(`/insights/${encodeURIComponent(city)}?country_code=${encodeURIComponent(cc)}`);
  const j = await res.json();

  if(!res.ok){
    el("statusBox").textContent = j.message || j.error || "No insights found.";
    return;
  }

  el("statusBox").textContent =
    `Status: ${j.status} | Updated: ${j.updated_at || "-"} | Run: ${j.analysis_run_id || "-"}` +
    (j.error ? ` | Error: ${j.error}` : "");

  if(j.status !== "ok" || !j.payload){
    return;
  }

  const p = j.payload;

  renderSummaryCards(p);
  renderMonthlyCharts(p);
  renderPatterns(p);
  renderExtremeTables(p);
}

el("rerunBtn").addEventListener("click", rerunAnalysis);
loadDashboard();