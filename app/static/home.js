const addCityBtn = document.getElementById("btnAddCity");
const forecastBtn = document.getElementById("btnForecast");
const insightsBtn = document.getElementById("btnInsights");

if (addCityBtn) {
  addCityBtn.addEventListener("click", () => {
    window.location.href = "/add-city";
  });
}

if (forecastBtn) {
  forecastBtn.addEventListener("click", () => {
    window.location.href = "/forecast-page";
  });
}

if (insightsBtn) {
  insightsBtn.addEventListener("click", () => {
    window.location.href = "/insights-page";
  });
}