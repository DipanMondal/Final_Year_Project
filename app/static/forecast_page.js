function $(id) {
  return document.getElementById(id);
}

let forecastChart = null;

function parseCityKey(raw) {
  if (!raw || typeof raw !== "string") {
    return { city: "", country_code: "" };
  }

  const trimmed = raw.trim();
  const parts = trimmed.split("_");

  if (parts.length >= 2) {
    const maybeCountry = parts[parts.length - 1];
    if (/^[a-zA-Z]{2}$/.test(maybeCountry)) {
      return {
        city: parts.slice(0, -1).join("_"),
        country_code: maybeCountry.toUpperCase(),
      };
    }
  }

  return {
    city: trimmed,
    country_code: "",
  };
}

function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}

function setStatus(message, kind = "") {
  const box = $("forecastStatus");
  box.className = "status-box" + (kind ? ` ${kind}` : "");
  box.textContent = message;
}

function fmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function card(label, value) {
  const div = document.createElement("div");
  div.className = "card-mini";
  div.innerHTML = `<div class="k">${label}</div><div class="v">${value}</div>`;
  return div;
}

function normalizeCitiesResponse(data) {
  if (Array.isArray(data)) return data;

  if (Array.isArray(data.cities)) return data.cities;
  if (Array.isArray(data.items)) return data.items;
  if (Array.isArray(data.results)) return data.results;

  return [];
}

function cityDisplayName(item) {
  if (typeof item === "string") return item;
  if (item.city_key) return item.city_key;
  if (item.city) return item.city;
  if (item.name) return item.name;
  return JSON.stringify(item);
}

function cityValue(item) {
  if (typeof item === "string") {
    return parseCityKey(item).city;
  }

  if (item.city) return item.city;

  if (item.city_key) {
    return parseCityKey(item.city_key).city;
  }

  if (item.name) return item.name;

  return "";
}

function inferCountryCode(item) {
  if (typeof item === "string") {
    return parseCityKey(item).country_code;
  }
  if (item.country_code) return String(item.country_code).toUpperCase();
  if (item.city_key) {
    return parseCityKey(item.city_key).country_code;
  }
  return "";
}

async function loadCities() {
  const select = $("city");
  select.innerHTML = `<option value="">Loading cities...</option>`;

  try {
    const res = await fetch("/cities");
    const data = await res.json().catch(() => ({}));
    const cities = normalizeCitiesResponse(data);

    select.innerHTML = `<option value="">Select a city</option>`;

    if (!cities.length) {
      select.innerHTML = `<option value="">No cities found</option>`;
      return;
    }

    cities.forEach((item) => {
	  const option = document.createElement("option");

	  const parsedCity = cityValue(item);
	  const parsedCountry = inferCountryCode(item);

	  option.value = parsedCity;
	  option.textContent = cityDisplayName(item); // keep DB-style label visible
	  option.dataset.countryCode = parsedCountry;

	  select.appendChild(option);
	});
  } catch (err) {
    select.innerHTML = `<option value="">Failed to load cities</option>`;
    setStatus(`Could not load city list: ${err.message}`, "error");
  }
}

function destroyChart() {
  if (forecastChart) {
    forecastChart.destroy();
    forecastChart = null;
  }
}

function renderForecastChart(predictions) {
  destroyChart();

  const canvas = $("forecastChart");
  const parent = canvas.parentElement;
  const rect = parent.getBoundingClientRect();

  if (!rect.width || !rect.height || rect.width > 5000 || rect.height > 5000) {
    return;
  }

  const labels = predictions.map((x) => x.date);
  const tavg = predictions.map((x) => Number(x.tavg));
  const lower = predictions.map((x) => Number(x.lower_95));
  const upper = predictions.map((x) => Number(x.upper_95));

  forecastChart = new Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Predicted tavg",
          data: tavg,
          tension: 0.25,
          borderWidth: 2,
        },
        {
          label: "Lower 95%",
          data: lower,
          tension: 0.2,
          borderDash: [6, 6],
          borderWidth: 1.5,
        },
        {
          label: "Upper 95%",
          data: upper,
          tension: 0.2,
          borderDash: [6, 6],
          borderWidth: 1.5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      resizeDelay: 120,
      plugins: {
        legend: {
          display: true,
        },
      },
      scales: {
        y: {
          ticks: {
            callback: (value) => `${value}°C`,
          },
        },
      },
    },
  });
}

function renderForecastTable(predictions) {
  const table = $("forecastTable");

  let html = `
    <tr>
      <th>Date</th>
      <th>Predicted tavg</th>
      <th>Lower 95%</th>
      <th>Upper 95%</th>
    </tr>
  `;

  predictions.forEach((row) => {
    html += `
      <tr>
        <td>${row.date ?? "-"}</td>
        <td>${fmt(row.tavg, 2)}</td>
        <td>${fmt(row.lower_95, 2)}</td>
        <td>${fmt(row.upper_95, 2)}</td>
      </tr>
    `;
  });

  table.innerHTML = html;
}

function renderForecastCards(data) {
  const cards = $("forecastCards");
  cards.innerHTML = "";

  const model = data.model_info || {};
  const preds = Array.isArray(data.predictions) ? data.predictions : [];

  const first = preds[0];
  const last = preds[preds.length - 1];

  cards.appendChild(card("City", data.city || "-"));
  cards.appendChild(card("Horizon", `${data.horizon_days ?? "-"} days`));
  cards.appendChild(card("CV MAE", fmt(model.cv_mae, 3)));
  cards.appendChild(card("Date range", first && last ? `${first.date} → ${last.date}` : "-"));
}

async function handleForecastSubmit(ev) {
  ev.preventDefault();

  const selectedOption = $("city").selectedOptions[0];
  let city = $("city").value.trim();
  let countryCode = $("country_code").value.trim().toUpperCase();
  const horizon = $("horizon").value;

  if (!city || !countryCode || !horizon) {
    setStatus("Please fill all fields.", "error");
    return;
  }

  const parsed = parseCityKey(city);
  if (parsed.country_code && parsed.country_code === countryCode) {
    city = parsed.city;
  }

  const submitBtn = $("forecastBtn");
  submitBtn.disabled = true;
  setStatus("Fetching forecast...", "loading");
  $("rawForecastBox").textContent = "{}";
  $("forecastCards").innerHTML = "";
  $("forecastTable").innerHTML = "";
  destroyChart();

  try {
    const url = `/forecast?city=${encodeURIComponent(city)}&country_code=${encodeURIComponent(countryCode)}&horizon=${encodeURIComponent(horizon)}`;
    const res = await fetch(url);
    const data = await res.json().catch(() => ({}));

    $("rawForecastBox").textContent = pretty(data);

    if (!res.ok) {
      setStatus(data.error || data.message || "Forecast request failed.", "error");
      return;
    }

    const predictions = Array.isArray(data.predictions) ? data.predictions : [];
    renderForecastCards(data);
    renderForecastTable(predictions);
    renderForecastChart(predictions);
    setStatus("Forecast loaded successfully.", "ok");
  } catch (err) {
    setStatus(`Network error: ${err.message}`, "error");
  } finally {
    submitBtn.disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  await loadCities();

  $("city").addEventListener("change", () => {
    const option = $("city").selectedOptions[0];
    const cc = option?.dataset?.countryCode || "";
    if (cc && !$("country_code").value.trim()) {
      $("country_code").value = cc;
    }
  });

  $("forecastForm").addEventListener("submit", handleForecastSubmit);
});