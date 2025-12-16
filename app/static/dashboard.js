function el(id){ return document.getElementById(id); }

function card(k, v){
  const d = document.createElement("div");
  d.className = "card";
  d.innerHTML = `<div class="k">${k}</div><div class="v">${v}</div>`;
  return d;
}

function tableFrom(rows, cols){
  let html = "<tr>" + cols.map(c => `<th>${c}</th>`).join("") + "</tr>";
  for(const r of rows){
    html += "<tr>" + cols.map(c => `<td>${r[c]}</td>`).join("") + "</tr>";
  }
  return html;
}

async function rerunAnalysis(){
  const city = window.DASH_CITY;
  const cc = window.DASH_COUNTRY || "";
  const start = window.DASH_START;
  const end = window.DASH_END;

  el("statusBox").textContent = "Re-running analysis... (this may take a bit)";
  const url = `/analyse/${encodeURIComponent(city)}?country_code=${encodeURIComponent(cc)}&start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`;
  const res = await fetch(url, { method: "POST" });
  const j = await res.json();
  if(!res.ok){
    el("statusBox").textContent = "Analysis failed: " + (j.error || "unknown");
    return;
  }
  await loadDashboard();
}

function plotLine(canvasId, labels, values, title){
  const ctx = el(canvasId);
  new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{ label: title, data: values, tension: 0.25 }]
    },
    options: { responsive: true }
  });
}

function plotBar(canvasId, labels, values, title){
  const ctx = el(canvasId);
  new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{ label: title, data: values }]
    },
    options: { responsive: true }
  });
}

async function loadDashboard(){
  const city = window.DASH_CITY;
  const cc = window.DASH_COUNTRY || "";
  el("title").textContent = `Weather Insights — ${city}${cc ? " (" + cc + ")" : ""}`;
  el("subtitle").textContent = `Range: ${window.DASH_START} → ${window.DASH_END}`;

  const res = await fetch(`/insights/${encodeURIComponent(city)}?country_code=${encodeURIComponent(cc)}`);
  const j = await res.json();

  if(!res.ok){
    el("statusBox").textContent = j.message || (j.error || "No insights found.");
    return;
  }

  el("statusBox").textContent =
    `Status: ${j.status} | Updated: ${j.updated_at || "-"} | Run: ${j.analysis_run_id || "-"}` +
    (j.error ? ` | Error: ${j.error}` : "");

  if(j.status !== "ok" || !j.payload){
    return;
  }

  const p = j.payload;

  // cards
  const cards = el("cards");
  cards.innerHTML = "";
  const s = p.summary || {};
  cards.appendChild(card("Annual trend (°C/year)", (s.annual_trend_c_per_year ?? 0).toFixed(3)));
  cards.appendChild(card("Anomaly days (|z| ≥ 2)", s.anomaly_days_count_abs_ge_2 ?? 0));

  const hm = s.hottest_month ? `${s.hottest_month.month_name} (${s.hottest_month.tavg_mean.toFixed(1)}°C)` : "-";
  const cm = s.coolest_month ? `${s.coolest_month.month_name} (${s.coolest_month.tavg_mean.toFixed(1)}°C)` : "-";
  cards.appendChild(card("Hottest month", hm));
  cards.appendChild(card("Coolest month", cm));

  // charts
  const mb = p.monthly_baseline || [];
  const labels = mb.map(x => x.month_name);
  plotLine("chartMean", labels, mb.map(x => x.tavg_mean), "Monthly mean tavg (°C)");
  plotBar("chartStd", labels, mb.map(x => x.tavg_std), "Monthly std tavg (°C)");
  plotLine("chartDiurnal", labels, mb.map(x => x.diurnal_mean), "Monthly diurnal mean (°C)");

  // clusters
  const clusters = el("clusters");
  clusters.innerHTML = "";
  const cls = (p.patterns && p.patterns.clusters) ? p.patterns.clusters : [];
  if(cls.length === 0){
    clusters.textContent = "No clusters found.";
  } else {
    for(const c of cls){
      const div = document.createElement("div");
      div.className = "cluster";
      const months = (c.months_names || []).join(", ");
      const yrs = (c.years || []).slice(0, 8).join(", ") + ((c.years || []).length > 8 ? "..." : "");
      const sig = (c.signature_top5 || []).map(x => `${x.feature}: ${x.direction}`).join(" | ");
      div.innerHTML = `
        <div><span class="badge">${c.label}</span></div>
        <div style="margin-top:8px;opacity:0.9">Months: ${months || "-"}</div>
        <div style="margin-top:6px;opacity:0.9">Years: ${yrs || "-"}</div>
        <div style="margin-top:6px;opacity:0.9">Signature: ${sig || "-"}</div>
      `;
      clusters.appendChild(div);
    }
  }

  // tables
  const ex = p.extremes || {};
  el("tblAnom").innerHTML = tableFrom((ex.top_anomaly_days || []).map(x => ({date:x.date, anomaly_z:x.anomaly_z.toFixed(2)})), ["date","anomaly_z"]);
  el("tblHot").innerHTML = tableFrom((ex.top_hot_days || []).map(x => ({date:x.date, tavg:x.tavg.toFixed(1)})), ["date","tavg"]);
  el("tblCold").innerHTML = tableFrom((ex.top_cold_days || []).map(x => ({date:x.date, tavg:x.tavg.toFixed(1)})), ["date","tavg"]);
}

el("rerunBtn").addEventListener("click", rerunAnalysis);
loadDashboard();
