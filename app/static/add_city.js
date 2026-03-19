function $(id) {
  return document.getElementById(id);
}

function setStatus(message, kind = "") {
  const box = $("statusBox");
  box.className = "status-box" + (kind ? ` ${kind}` : "");
  box.textContent = message;
}

function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}

function buildPayload() {
  return {
    city: $("city").value.trim(),
    country_code: $("country_code").value.trim().toUpperCase(),
    start: $("start").value,
    end: $("end").value,
  };
}

function refreshPreview() {
  $("requestPreview").textContent = pretty(buildPayload());
}

async function handleSubmit(ev) {
  ev.preventDefault();

  const payload = buildPayload();

  if (!payload.city || !payload.country_code || !payload.start || !payload.end) {
    setStatus("Please fill all fields.", "error");
    return;
  }

  if (payload.start > payload.end) {
    setStatus("Start date must be earlier than or equal to end date.", "error");
    return;
  }

  const submitBtn = $("submitBtn");
  submitBtn.disabled = true;
  setStatus("Submitting city ingestion request...", "loading");
  $("responseBox").textContent = "{}";

  try {
    const res = await fetch("/cities", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const data = await res.json().catch(() => ({}));
    $("responseBox").textContent = pretty(data);

    if (!res.ok) {
      setStatus(data.error || data.message || "Request failed.", "error");
      return;
    }

    setStatus("City ingestion request completed successfully.", "ok");
  } catch (err) {
    setStatus(`Network error: ${err.message}`, "error");
  } finally {
    submitBtn.disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const today = new Date().toISOString().slice(0, 10);
  if (!$("end").value) $("end").value = today;
  if (!$("start").value) $("start").value = "2023-01-01";

  ["city", "country_code", "start", "end"].forEach((id) => {
    $(id).addEventListener("input", refreshPreview);
    $(id).addEventListener("change", refreshPreview);
  });

  $("addCityForm").addEventListener("submit", handleSubmit);
  $("resetBtn").addEventListener("click", () => {
    setTimeout(refreshPreview, 0);
    $("responseBox").textContent = "{}";
    setStatus("Waiting for submission.");
  });

  refreshPreview();
});