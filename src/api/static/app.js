const elements = {
  form: document.querySelector("#prediction-form"),
  ticker: document.querySelector("#ticker"),
  button: document.querySelector("#predict-button"),
  status: document.querySelector("#api-status"),
  resultCard: document.querySelector("#result-card"),
  errorCard: document.querySelector("#error-card"),
  forecastValue: document.querySelector("#forecast-value"),
  resultTicker: document.querySelector("#result-ticker"),
  asOfDate: document.querySelector("#as-of-date"),
  errorMessage: document.querySelector("#error-message"),
  modelName: document.querySelector("#model-name"),
  modelVersion: document.querySelector("#model-version"),
  featureCount: document.querySelector("#feature-count"),
  forecastHorizon: document.querySelector("#forecast-horizon"),
};

function showError(message) {
  elements.resultCard.classList.add("is-hidden");
  elements.errorMessage.textContent = message;
  elements.errorCard.classList.remove("is-hidden");
}

function clearError() {
  elements.errorCard.classList.add("is-hidden");
  elements.errorMessage.textContent = "";
}

function formatDate(value) {
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  }).format(new Date(`${value}T00:00:00Z`));
}

function messageForResponse(status, detail) {
  if (status === 404) {
    return "No stored market data exists for this ticker yet.";
  }
  if (status === 422) {
    return detail || "This ticker needs more history or matching SPY/VIX market context.";
  }
  if (status === 503) {
    return "The forecast service is temporarily unavailable. Please try again shortly.";
  }
  return detail || "The forecast could not be completed. Please try again.";
}

async function readJson(response) {
  try {
    return await response.json();
  } catch {
    return {};
  }
}

async function loadModel() {
  try {
    const response = await fetch("/v1/model");
    const model = await readJson(response);
    if (!response.ok) {
      throw new Error(model.detail || "The model information is unavailable.");
    }

    elements.ticker.replaceChildren();
    for (const ticker of model.supported_tickers) {
      const option = document.createElement("option");
      option.value = ticker;
      option.textContent = ticker;
      elements.ticker.append(option);
    }
    elements.modelName.textContent = model.model_name;
    elements.modelVersion.textContent = model.model_version;
    elements.featureCount.textContent = `${model.feature_count} engineered inputs`;
    elements.forecastHorizon.textContent = `${model.forecast_horizon_trading_days} trading days`;
    elements.ticker.disabled = false;
    elements.button.disabled = false;
    elements.status.textContent = "Model ready";
    elements.status.classList.add("is-ready");
  } catch (error) {
    const message = error instanceof Error ? error.message : "The model information is unavailable.";
    elements.status.textContent = "API unavailable";
    showError(message);
  }
}

async function requestPrediction(event) {
  event.preventDefault();
  clearError();
  elements.resultCard.classList.add("is-hidden");
  elements.button.disabled = true;
  elements.button.textContent = "Calculating…";

  try {
    const ticker = elements.ticker.value;
    const response = await fetch(`/v1/predictions/${encodeURIComponent(ticker)}`, {
      method: "POST",
    });
    const prediction = await readJson(response);
    if (!response.ok) {
      showError(messageForResponse(response.status, prediction.detail));
      return;
    }

    elements.forecastValue.textContent = `${(prediction.predicted_rv_20d * 100).toFixed(2)}%`;
    elements.resultTicker.textContent = prediction.ticker;
    elements.asOfDate.textContent = formatDate(prediction.as_of_date);
    elements.resultCard.classList.remove("is-hidden");
  } catch {
    showError("The API could not be reached. Check that the service is running and try again.");
  } finally {
    elements.button.disabled = false;
    elements.button.textContent = "Predict volatility";
  }
}

elements.form.addEventListener("submit", requestPrediction);
loadModel();
