const elements = {
  form: document.querySelector("#prediction-form"),
  ticker: document.querySelector("#ticker"),
  button: document.querySelector("#predict-button"),
  status: document.querySelector("#api-status"),
  resultCard: document.querySelector("#result-card"),
  errorCard: document.querySelector("#error-card"),
  forecastPredictions: document.querySelector("#forecast-predictions"),
  resultTicker: document.querySelector("#result-ticker"),
  asOfDate: document.querySelector("#as-of-date"),
  marketSummary: document.querySelector("#market-summary"),
  snapshotAsOfDate: document.querySelector("#snapshot-as-of"),
  snapshotPrice: document.querySelector("#snapshot-price"),
  snapshotReturn1d: document.querySelector("#snapshot-return-1d"),
  snapshotReturn5d: document.querySelector("#snapshot-return-5d"),
  snapshotReturn20d: document.querySelector("#snapshot-return-20d"),
  snapshotRv20d: document.querySelector("#snapshot-rv-20d"),
  snapshotDrawdown: document.querySelector("#snapshot-drawdown"),
  snapshotVolumeRatio: document.querySelector("#snapshot-volume-ratio"),
  errorMessage: document.querySelector("#error-message"),
  modelName: document.querySelector("#model-name"),
  modelVersion: document.querySelector("#model-version"),
  modelTarget: document.querySelector("#model-target"),
  featureCount: document.querySelector("#feature-count"),
  forecastHorizon: document.querySelector("#forecast-horizon"),
  modelInput: document.querySelector("#model-input"),
  modelSelector: document.querySelector("#model-selector"),
  rangeButtons: document.querySelectorAll(".range-button"),
  historyButton: document.querySelector("#history-button"),
  marketHistory: document.querySelector("#market-history"),
  priceChart: document.querySelector("#price-chart"),
  volatilityChart: document.querySelector("#volatility-chart"),
  historySummary: document.querySelector("#history-summary"),
  historyTableBody: document.querySelector("#history-table-body"),
};

const historyState = { range: "1y" };
const modelState = { championModelId: "", models: new Map() };
const svgNamespace = "http://www.w3.org/2000/svg";

function showError(message, hideForecast = true) {
  if (hideForecast) {
    elements.resultCard.classList.add("is-hidden");
  }
  elements.errorMessage.textContent = message;
  elements.errorCard.classList.remove("is-hidden");
}

function clearError() {
  elements.errorCard.classList.add("is-hidden");
  elements.errorMessage.textContent = "";
}

function setHistoryRange(range) {
  historyState.range = range;
  for (const button of elements.rangeButtons) {
    const selected = button.dataset.range === range;
    button.classList.toggle("is-selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  }
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

    const champion = model.models.find((item) => item.model_id === model.champion_model_id);
    if (!champion) {
      throw new Error("The API did not identify a champion model.");
    }

    modelState.championModelId = model.champion_model_id;
    modelState.models = new Map(model.models.map((item) => [item.model_id, item]));
    elements.ticker.replaceChildren();
    for (const ticker of champion.supported_tickers) {
      const option = document.createElement("option");
      option.value = ticker;
      option.textContent = ticker;
      elements.ticker.append(option);
    }
    elements.modelSelector.replaceChildren();
    for (const item of model.models) {
      const option = document.createElement("option");
      option.value = item.model_id;
      option.textContent = item.model_id === model.champion_model_id
        ? `${item.display_name} · Champion`
        : `${item.display_name} · Comparison`;
      elements.modelSelector.append(option);
    }
    elements.modelSelector.value = model.champion_model_id;
    renderModelDetails(champion);
    elements.ticker.disabled = false;
    elements.button.disabled = false;
    elements.modelSelector.disabled = false;
    elements.historyButton.disabled = false;
    for (const button of elements.rangeButtons) {
      button.disabled = false;
    }
    elements.status.textContent = "Model ready";
    elements.status.classList.add("is-ready");
  } catch (error) {
    const message = error instanceof Error ? error.message : "The model information is unavailable.";
    elements.status.textContent = "API unavailable";
    showError(message);
  }
}

function renderModelDetails(model) {
  elements.modelName.textContent = model.display_name;
  elements.modelVersion.textContent = model.model_version;
  elements.modelTarget.textContent = model.target_name;
  elements.featureCount.textContent = `${model.feature_count} engineered inputs`;
  elements.forecastHorizon.textContent = `${model.forecast_horizon_trading_days} trading days`;
  elements.modelInput.textContent = model.input_requirement;
}

function createSvgElement(name, attributes = {}) {
  const element = document.createElementNS(svgNamespace, name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, String(value));
  }
  return element;
}

function renderLineChart(svg, points, field, valueFormatter) {
  svg.replaceChildren();
  const width = 680;
  const height = 240;
  const padding = { top: 18, right: 16, bottom: 32, left: 64 };
  const values = points.map((point) => point[field]);
  let minimum = Math.min(...values);
  let maximum = Math.max(...values);
  if (minimum === maximum) {
    minimum -= Math.max(Math.abs(minimum) * 0.05, 0.01);
    maximum += Math.max(Math.abs(maximum) * 0.05, 0.01);
  }
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;
  const xFor = (index) => padding.left + (index / Math.max(points.length - 1, 1)) * chartWidth;
  const yFor = (value) => padding.top + ((maximum - value) / (maximum - minimum)) * chartHeight;

  for (const ratio of [0, 0.5, 1]) {
    const y = padding.top + ratio * chartHeight;
    svg.append(createSvgElement("line", {
      class: "chart-grid-line", x1: padding.left, x2: width - padding.right, y1: y, y2: y,
    }));
  }
  const path = points.map((point, index) => `${index === 0 ? "M" : "L"}${xFor(index)} ${yFor(point[field])}`).join(" ");
  svg.append(createSvgElement("path", { class: "chart-line", d: path }));

  const labels = [
    { text: valueFormatter(maximum), x: 0, y: padding.top + 4, anchor: "start" },
    { text: valueFormatter(minimum), x: 0, y: height - padding.bottom + 4, anchor: "start" },
    { text: formatDate(points[0].date), x: padding.left, y: height - 5, anchor: "start" },
    { text: formatDate(points.at(-1).date), x: width - padding.right, y: height - 5, anchor: "end" },
  ];
  for (const label of labels) {
    const text = createSvgElement("text", {
      class: "chart-label", x: label.x, y: label.y, "text-anchor": label.anchor,
    });
    text.textContent = label.text;
    svg.append(text);
  }
}

function formatNumber(value, maximumFractionDigits = 2) {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits }).format(value);
}

function formatPercent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function renderSignedPercent(element, value) {
  element.textContent = `${value > 0 ? "+" : ""}${formatPercent(value)}`;
  element.classList.toggle("metric-positive", value > 0);
  element.classList.toggle("metric-negative", value < 0);
}

function renderForecasts(prediction) {
  elements.forecastPredictions.replaceChildren();
  for (const item of prediction.predictions) {
    const card = document.createElement("article");
    card.className = "forecast-model-card";
    if (item.model_id === prediction.champion_model_id) {
      card.classList.add("is-champion");
    }

    const label = document.createElement("p");
    label.className = "forecast-model-label";
    label.textContent = item.model_id === prediction.champion_model_id
      ? `${item.display_name} · Champion`
      : `${item.display_name} · Comparison`;
    const value = document.createElement("p");
    value.className = "forecast-value";
    value.textContent = formatPercent(item.predicted_rv_20d);
    const version = document.createElement("p");
    version.className = "forecast-model-version";
    version.textContent = `Version ${item.model_version}`;
    card.append(label, value, version);
    elements.forecastPredictions.append(card);
  }
}

function renderMarketSummary(summary) {
  elements.snapshotAsOfDate.textContent = `As of ${formatDate(summary.as_of_date)}`;
  elements.snapshotPrice.textContent = `$${formatNumber(summary.adjusted_close)}`;
  renderSignedPercent(elements.snapshotReturn1d, summary.return_1d);
  renderSignedPercent(elements.snapshotReturn5d, summary.return_5d);
  renderSignedPercent(elements.snapshotReturn20d, summary.return_20d);
  elements.snapshotRv20d.textContent = formatPercent(summary.rv_20d);
  elements.snapshotDrawdown.textContent = formatPercent(summary.drawdown);
  elements.snapshotVolumeRatio.textContent = `${formatNumber(summary.volume_ratio_20d)}x`;
  elements.marketSummary.classList.remove("is-hidden");
}

async function loadMarketSummary(ticker) {
  elements.marketSummary.classList.add("is-hidden");

  try {
    const response = await fetch(`/v1/market-summary/${encodeURIComponent(ticker)}`);
    const summary = await readJson(response);
    if (!response.ok) {
      showError(messageForResponse(response.status, summary.detail), false);
      return;
    }
    renderMarketSummary(summary);
  } catch {
    showError("The forecast succeeded, but the market snapshot could not be loaded.", false);
  }
}

function renderHistoryTable(points) {
  elements.historyTableBody.replaceChildren();
  for (const point of points.slice(-20).reverse()) {
    const row = document.createElement("tr");
    const cells = [
      formatDate(point.date),
      formatNumber(point.adjusted_close),
      formatPercent(point.rv_20d),
      formatPercent(point.return_20d),
      formatPercent(point.drawdown),
      `${formatNumber(point.volume_ratio_20d)}x`,
    ];
    for (const value of cells) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    elements.historyTableBody.append(row);
  }
}

async function loadMarketHistory() {
  clearError();
  elements.historyButton.disabled = true;
  elements.historyButton.textContent = "Loading history…";

  try {
    const ticker = elements.ticker.value;
    const response = await fetch(
      `/v1/market-data/${encodeURIComponent(ticker)}?range=${historyState.range}`,
    );
    const history = await readJson(response);
    if (!response.ok) {
      showError(messageForResponse(response.status, history.detail));
      return;
    }
    renderLineChart(elements.priceChart, history.points, "adjusted_close", (value) => formatNumber(value));
    renderLineChart(elements.volatilityChart, history.points, "rv_20d", formatPercent);
    renderHistoryTable(history.points);
    elements.historySummary.textContent = `${history.points.length} of ${history.available_observations} causal rows`;
    elements.marketHistory.classList.remove("is-hidden");
  } catch {
    showError("Market history could not be loaded. Check the API and try again.");
  } finally {
    elements.historyButton.disabled = false;
    elements.historyButton.textContent = "Load market history";
  }
}

async function requestPrediction(event) {
  event.preventDefault();
  clearError();
  elements.resultCard.classList.add("is-hidden");
  elements.marketSummary.classList.add("is-hidden");
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

    renderForecasts(prediction);
    elements.resultTicker.textContent = `${prediction.company_name} (${prediction.ticker})`;
    elements.asOfDate.textContent = formatDate(prediction.as_of_date);
    elements.resultCard.classList.remove("is-hidden");
    await loadMarketSummary(prediction.ticker);
  } catch {
    showError("The API could not be reached. Check that the service is running and try again.");
  } finally {
    elements.button.disabled = false;
    elements.button.textContent = "Predict volatility";
  }
}

elements.form.addEventListener("submit", requestPrediction);
elements.modelSelector.addEventListener("change", () => {
  const selected = modelState.models.get(elements.modelSelector.value);
  if (selected) {
    renderModelDetails(selected);
  }
});
for (const button of elements.rangeButtons) {
  button.addEventListener("click", () => setHistoryRange(button.dataset.range));
}
elements.historyButton.addEventListener("click", loadMarketHistory);
loadModel();
