function $(id) {
  return document.getElementById(id);
}

function setStatus(message, kind = "") {
  const box = $("statusBox");
  box.className = "status-box" + (kind ? ` ${kind}` : "");
  box.textContent = message;
}

function normalizeCitiesResponse(data) {
  if (Array.isArray(data)) return data;
  if (Array.isArray(data.cities)) return data.cities;
  if (Array.isArray(data.items)) return data.items;
  if (Array.isArray(data.results)) return data.results;
  return [];
}

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

function cityDisplayName(item) {
  if (typeof item === "string") {
    const parsed = parseCityKey(item);
    return parsed.country_code ? `${parsed.city} (${parsed.country_code})` : parsed.city;
  }

  if (item.city && item.country_code) {
    return `${item.city} (${String(item.country_code).toUpperCase()})`;
  }

  if (item.city_key) {
    const parsed = parseCityKey(item.city_key);
    return parsed.country_code ? `${parsed.city} (${parsed.country_code})` : parsed.city;
  }

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

function refreshPreview() {
  const city = $("city").value.trim();
  $("routePreview").textContent = city ? `/dashboard/${city}` : "/dashboard/";
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
      setStatus("No cities available yet. Add a city first.", "error");
      return;
    }

    cities.forEach((item) => {
      const option = document.createElement("option");
      option.value = cityValue(item);
      option.textContent = cityDisplayName(item);
      select.appendChild(option);
    });

    setStatus("City list loaded.", "ok");
  } catch (err) {
    select.innerHTML = `<option value="">Failed to load cities</option>`;
    setStatus(`Could not load city list: ${err.message}`, "error");
  }
}

function openDashboard(ev) {
  ev.preventDefault();

  const city = $("city").value.trim();
  if (!city) {
    setStatus("Please select a city.", "error");
    return;
  }

  const url = `/dashboard/${encodeURIComponent(city)}`;
  setStatus("Opening dashboard...", "loading");
  window.location.href = url;
}

document.addEventListener("DOMContentLoaded", async () => {
  await loadCities();
  $("city").addEventListener("change", refreshPreview);
  $("insightsForm").addEventListener("submit", openDashboard);
  refreshPreview();
});