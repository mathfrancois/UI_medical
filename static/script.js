let lastTrainedModelPath = null;
let lastUsedTargetColumn = null;
let lastCleanedCsvBlob = null;
let summaryResults = {};
let dataSetPreview = null;
let shapPlot = null;

window.addEventListener("DOMContentLoaded", () => {
  const savedLang = localStorage.getItem("selectedLanguage") || "en";
  document.getElementById("language-select").value = savedLang;
  changeLanguage(savedLang);
});

function updateTimeDisplay() {
    const slider = document.getElementById("training-time-limit");
    const display = document.getElementById("time-limit-display");
    display.textContent = slider.value;
}

const settingsToggle = document.getElementById("settings-toggle");
const settingsMenu = document.getElementById("settings-menu");

settingsToggle.addEventListener("click", () => {
  settingsMenu.classList.toggle("hidden");
});

// Fermer si on clique en dehors
document.addEventListener("click", (event) => {
  if (
    !settingsMenu.contains(event.target) &&
    !settingsToggle.contains(event.target)
  ) {
    settingsMenu.classList.add("hidden");
  }
});

function changeLanguage(lang) {
  if (!lang) {
    lang = document.getElementById("language-select").value;
  }

  // Mémoriser la langue choisie
  localStorage.setItem("selectedLanguage", lang);

  // Appliquer les traductions
  const elements = document.querySelectorAll("[data-i18n]");
  elements.forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (translations[lang] && translations[lang][key]) {
      el.textContent = translations[lang][key];
    }
  });

  // Placeholder spécial
  const chatInput = document.getElementById("chat-input");
  if (translations[lang]["chatPlaceholder"]) {
    chatInput.placeholder = translations[lang]["chatPlaceholder"];
  }
}

function t(key) {
  const lang = localStorage.getItem("selectedLanguage") || "en";
  return translations[lang] && translations[lang][key] ? translations[lang][key] : key;
}


function switchMode() {
  const mode = document.querySelector('input[name="mode"]:checked').value;

  document.getElementById('training-section').style.display = mode === 'train' ? 'block' : 'none';
  document.getElementById('predict-section').style.display = mode === 'predict' ? 'block' : 'none';
}



// Appel initial au chargement
document.addEventListener("DOMContentLoaded", updateTimeDisplay);

const timeLimitSeconds = parseInt(document.getElementById("training-time-limit").value);

function toggleLoadChoice() {
  const choice = document.querySelector('input[name="load-choice"]:checked').value;
  document.getElementById('csv-upload-section').style.display = (choice === 'dataset') ? 'block' : 'none';
  document.getElementById('model-upload-section').style.display = (choice === 'model') ? 'block' : 'none';
}

function removeModelFile() {
  const input = document.getElementById('upload-model');
  input.value = '';
  input.style.display = 'inline';
  document.getElementById('model-file-info').style.display = 'none';
  document.getElementById('model-file-name').textContent = '';
  document.getElementById('model-status').style.display = 'none';
}


document.getElementById('upload-csv').addEventListener('change', function () {
  const file = this.files[0];
  const acceptedExtensions = ['.csv', '.xls', '.xlsx', '.xlsm', '.arff'];
  if (file && acceptedExtensions.some(ext => file.name.endsWith(ext))) {
    // Cacher input + bouton
    this.style.display = 'none';
    document.getElementById('upload-csv-label').style.display = 'none';

    document.getElementById('csv-file-name').textContent = file.name;
    document.getElementById('csv-file-info').style.display = 'inline';

    const previewTable = document.getElementById('preview-table');
    previewTable.innerHTML = ''; // Reset tableau
    document.getElementById('csv-preview').style.display = 'block';

    const ext = file.name.split('.').pop().toLowerCase();
    const reader = new FileReader();

    if (ext === 'csv' || ext === 'arff') {
      reader.onload = function (e) {
        const content = e.target.result;
        const parsed = Papa.parse(content, {
          header: false,
          skipEmptyLines: true
        });
        buildPreview(parsed.data);
      };
      reader.readAsText(file);
    } else if (['xls', 'xlsx', 'xlsm'].includes(ext)) {
      reader.onload = function (e) {
        const data = new Uint8Array(e.target.result);
        const workbook = XLSX.read(data, { type: 'array' });
        const sheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[sheetName];
        const json = XLSX.utils.sheet_to_json(worksheet, { header: 1, defval: "" });
        buildPreview(json);
      };
      reader.readAsArrayBuffer(file);
    }
    
  } else {
    showAlert(t("pleaseSelectFile"), 'warning');
    this.value = '';
  }
});

function buildStats(rows) {
  const header = rows[0];
  const dataRows = rows.slice(1);

  const stats = header.map((col, idx) => {
    const colData = dataRows.map(row => row[idx]).filter(val => val !== "" && val !== null && val !== undefined);
    const isNumeric = colData.every(val => !isNaN(parseFloat(val)));

    const parsedData = colData.map(val => isNumeric ? parseFloat(val) : val);
    const missing = dataRows.length - colData.length;

    const stat = {
      name: col,
      type: isNumeric ? "Numeric" : "Categorical",
      missing: missing,
      mean: isNumeric ? (parsedData.reduce((a, b) => a + b, 0) / parsedData.length).toFixed(2) : "-",
      std: isNumeric
        ? Math.sqrt(parsedData.map(x => (x - parsedData.reduce((a, b) => a + b, 0) / parsedData.length) ** 2).reduce((a, b) => a + b, 0) / parsedData.length).toFixed(2)
        : "-",
    };
    return stat;
  });

  displayStatsTable(stats);
}


function displayStatsTable(stats) {
  const table = document.getElementById('stats-table');
  table.innerHTML = '';

  const headerRow = document.createElement('tr');
  const headers = [
    { key: 'statsColumn' },
    { key: 'statsType' },
    { key: 'statsMissing' },
    { key: 'statsMean' },
    { key: 'statsStd' }
  ];

  headers.forEach(header => {
    const th = document.createElement('th');
    th.setAttribute('data-i18n', header.key);
    th.textContent = t(header.key);
    headerRow.appendChild(th);
  });

  table.appendChild(headerRow);

  stats.forEach(stat => {
    const row = document.createElement('tr');
    [stat.name, stat.type, stat.missing, stat.mean, stat.std || '-'].forEach(val => {
      const td = document.createElement('td');
      td.textContent = val;
      row.appendChild(td);
    });
    table.appendChild(row);
  });
}




function buildPreview(rows) {
  const previewTable = document.getElementById('preview-table');
  previewTable.innerHTML = '';

  const header = rows[0];
  const numColumns = header.length;
  const numRows = rows.length - 1;

  let nanCount = 0;
  for (let i = 1; i < rows.length; i++) {
    nanCount += rows[i].filter(cell =>
      cell === null || cell === undefined || cell.toString().trim() === "" || cell.toString().toLowerCase() === "nan"
    ).length;
  }

  // Stats
  document.getElementById('csv-rows-count').textContent = numRows;
  document.getElementById('csv-columns-count').textContent = numColumns;
  document.getElementById('csv-nan-count').textContent = nanCount;

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  header.forEach(col => {
    const th = document.createElement("th");
    th.textContent = col;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  previewTable.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (let i = 1; i < Math.min(rows.length, 6); i++) {
    const row = rows[i];
    const tr = document.createElement("tr");
    row.forEach(cell => {
      const td = document.createElement("td");
      td.textContent = cell;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  }
  previewTable.appendChild(tbody);

  // Mise à jour target-column
  const targetSelect = document.getElementById("target-column");
  targetSelect.innerHTML = "";
  header.forEach(col => {
    const option = document.createElement("option");
    option.value = col;
    option.textContent = col;
    targetSelect.appendChild(option);
  });
  buildStats(rows);
  const dropListContainer = document.getElementById("drop-columns-list");
  dropListContainer.innerHTML = "";

  header.forEach(col => {
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = col;
    checkbox.id = `drop-${col}`;

    const label = document.createElement("label");
    label.setAttribute("for", `drop-${col}`);
    label.textContent = col;
    label.style.marginRight = "15px";

    const container = document.createElement("div");
    container.appendChild(checkbox);
    container.appendChild(label);

    dropListContainer.appendChild(container);
  });
}


function removeCSVFile() {
  const input = document.getElementById('upload-csv');
  input.value = '';
  document.getElementById('upload-csv-label').style.display = 'inline-block';

  document.getElementById('csv-file-info').style.display = 'none';
  document.getElementById('csv-file-name').textContent = '';
  document.getElementById('csv-preview').style.display = 'none';
  document.getElementById('preview-table').innerHTML = '';
  document.getElementById('target-column').innerHTML = '';
  
  const resultsDiv = document.getElementById('training-results');
  resultsDiv.style.display = 'none';
  dataSetPreview = null;
}

function startTraining() {
  const fileInput = document.getElementById('upload-csv');
  const file = fileInput.files[0];
  const targetColumn = document.getElementById('target-column').value;

  if (!file || !targetColumn) {
    showAlert(t("selectCSVAndTarget"), 'warning');
    return;
  }

  // Colonnes cochées à supprimer
  const checkboxes = document.querySelectorAll('#drop-columns-list input[type="checkbox"]:checked');
  const columnsToDrop = Array.from(checkboxes).map(cb => cb.value);

  if (columnsToDrop.includes(targetColumn)) {
    showAlert(t("cannotDropTargetColumn"), 'error');
    return;
  }

  document.getElementById('training-spinner').style.display = "inline-block";

  const ext = file.name.split('.').pop().toLowerCase();
  const reader = new FileReader();

  reader.onload = function (e) {
    let data = [];
    let header = [];

    if (ext === 'csv' || ext === 'arff') {
      const parsed = Papa.parse(e.target.result, { header: false, skipEmptyLines: false });
      data = parsed.data;
    } else if (['xls', 'xlsx', 'xlsm'].includes(ext)) {
      const workbook = XLSX.read(new Uint8Array(e.target.result), { type: 'array' });
      const sheetName = workbook.SheetNames[0];
      const worksheet = workbook.Sheets[sheetName];
      data = XLSX.utils.sheet_to_json(worksheet, { header: 1, defval: "" });
    } else {
      showAlert(t("unsupportedFormat"), 'warning');
      document.getElementById('training-spinner').style.display = "none";
      return;
    }

    // Supprimer les colonnes cochées
    header = data[0];
    const dropIndexes = header.map((name, idx) => columnsToDrop.includes(name) ? idx : -1).filter(i => i !== -1);
    const cleanedData = data.map(row => row.filter((_, idx) => !dropIndexes.includes(idx)));

    // Reconversion en CSV
    const csv = Papa.unparse(cleanedData, { quotes: false });

    // Préparation du fichier nettoyé à envoyer
    const formData = new FormData();
    const blob = new Blob([csv], { type: 'text/csv' });
    formData.append('file', blob, `${file.name}.csv`);
    formData.append('target_column', targetColumn);
    formData.append('time_limit', timeLimitSeconds);

    lastUsedTargetColumn = targetColumn;
    lastCleanedCsvBlob = blob;  

    // Envoi au serveur
    fetch('/train', {
      method: 'POST',
      body: formData
    })
    .then(response => response.json())
    .then(data => {
      document.getElementById('training-spinner').style.display = "none";
      if (data.error) {
        showAlert(t(data.error), 'error');
        return;
      }
      renderTrainingResults(data);
    })
    .catch(error => {
      console.log(error);
      document.getElementById('training-spinner').style.display = "none";
      showAlert(t("trainingError"), 'error');
    });
  };

  if (ext === 'csv' || ext === 'arff') {
    reader.readAsText(file);
  } else {
    reader.readAsArrayBuffer(file);
  }
}


function renderTrainingResults(data) {
  lastTrainedModelPath = data.model_path;
  summaryResults["summary"] = data.summary_LLM;
  summaryResults["feature_importance_plot"] = data.feature_importance_plot;
  summaryResults["metrics_plot"] = data.metrics;
  dataSetPreview = data.markdown_preview; 
  const resultsDiv = document.getElementById('training-results');
  resultsDiv.innerHTML = `<h2 data-i18n="trainingResults">${t("trainingResults")}</h2>`;

  // -------- Training Summary Section --------
  let summaryHTML = `
    <div class="result-section" id="training-summary-section">
      <h3 data-i18n="trainingSummary">${t("trainingSummary")}</h3>

      <div class="subsection">
        <h4 data-i18n="generalInfo">${t("generalInfo")}</h4>
        <p><strong data-i18n="detectedTask">${t("detectedTask")}</strong> ${data.task_type}</p>
        <p><strong data-i18n="selectedModel">${t("selectedModel")}</strong> ${data.best_model}</p>
        <p><strong data-i18n="trainingTime">${t("trainingTime")}</strong> ${data.train_time.toFixed(2)} seconds</p>
      </div>
  `;

  const metrics = data.metrics;
  let metricsHTML = '';
  let plotsHTML = `
      <div class="subsection">
        <h4 data-i18n="plots">${t("plots")}</h4>
  `;

  for (const metric in metrics) {
    const value = metrics[metric].value;
    const plotBase64 = metrics[metric].plot;
    const plotHist = metrics[metric].plot_hist || null;
    const metricLabel = metric.toUpperCase();
    const formattedValue = value.toFixed(4);

    metricsHTML += `<tr><td>${metricLabel}</td><td>${formattedValue}</td></tr>`;

    if (metricLabel === "RMSE" && plotBase64 && plotHist) {
      plotsHTML += `
        <div class="plot-card">
          <p><strong>${metricLabel}</strong></p>
          <img src="data:image/png;base64,${plotBase64}" alt="${metricLabel} Plot" />
          <img src="data:image/png;base64,${plotHist}" alt="${metricLabel} Histogram" />
        </div>
      `;
    } else if (plotBase64) {
      plotsHTML += `
        <div class="plot-card">
          <p><strong>${metricLabel}</strong></p>
          <img src="data:image/png;base64,${plotBase64}" alt="${metricLabel} Plot" />
        </div>
      `;
    }
  }

  const metricSectionTitle = data.task_type === "regression" ? "Regression Metrics" : "Classification Metrics";

  summaryHTML += `
      <div class="subsection">
        <h4>${metricSectionTitle}</h4>
        <table class="metrics-table">
          <thead><tr><th>${t("metric")}</th><th>${t("value")}</th></tr></thead>
          <tbody>${metricsHTML}</tbody>
        </table>
      </div>
  `;

    if (data.leaderboard && Array.isArray(data.leaderboard)) {
    const leaderboardHTML = `
      <div class="result-section" id="leaderboard-section">
        <h3 data-i18n="leaderboard">${t("leaderboard")}</h3>
        <div class="leaderboard-table-container">
          <table class="metrics-table">
            <thead>
              <tr>
                <th data-i18n="model">${t("model")}</th>
                <th data-i18n="scoreVal">${t("scoreVal")}</th>
                <th data-i18n="fitTime">${t("fitTime")}</th>
                <th data-i18n="predictTime">${t("predictTime")}</th>
              </tr>
            </thead>
            <tbody>
              ${data.leaderboard.map(row => `
                <tr>
                  <td>${row.model}</td>
                  <td>${row.score_val.toFixed(4)}</td>
                  <td>${row.fit_time.toFixed(2)}s</td>
                  <td>${row.pred_time_val.toFixed(2)}s</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      </div>
    `;
    summaryHTML += leaderboardHTML;
  }

  plotsHTML += `</div>`; // Fin plots
  summaryHTML += plotsHTML + `</div>`; // Fin Training Summary
  resultsDiv.innerHTML += summaryHTML;

  // -------- Model Explainability Section --------
  let explainabilityHTML = `
    <div class="result-section" id="model-explainability-section">
      <h3 data-i18n="modelExplainability">${t("modelExplainability")}</h3>
  `;

  if (data.feature_importance_plot) {
    explainabilityHTML += `
      <div class="plot-card">
        <h4 data-i18n="featureImportance">${t("featureImportance")}</h4>
        <img src="data:image/png;base64,${data.feature_importance_plot}" alt="Feature Importance Plot" />
      </div>
    `;
  }

  explainabilityHTML += `
  <div class="button-container">
    <button id="generate-shap-button" onclick="generateShapPlot()" class="generate-shap-button" data-i18n="generateShap">
      ${t("generateShap")}
    </button>
    <div id="training-spinner-shap" class="spinner" style="display: none;"></div>
  </div>
  <div class="plot-card" id="shap-plots"></div>`;

  explainabilityHTML += `</div>`; // Fin Model Explainability
  resultsDiv.innerHTML += explainabilityHTML;

  // -------- Download Buttons --------
  resultsDiv.innerHTML += `
    <div class="download-buttons-container">
      <a id="download-link" href="#" data-i18n="downloadModel" class="download-button download-model" download>
        🧠 ${t("downloadModel")}
      </a>
      <button class="download-button download-plots" onclick="downloadAllPlots()" data-i18n="downloadPlots">
        📊 ${t("downloadPlots")}
      </button>
    </div>
  `;

  if (data.download_url) {
    document.getElementById('download-link').href = data.download_url;
  }

  resultsDiv.style.display = 'block';
}

async function generateShapPlot() {
  if (!lastTrainedModelPath || !lastCleanedCsvBlob || !lastUsedTargetColumn) {
    showAlert(t("missingSHAPInfo"), 'error');
    return;
  }
  document.getElementById('training-spinner-shap').style.display = "inline-block";

  const formData = new FormData();
  formData.append("model_path", lastTrainedModelPath);
  formData.append("target_column", lastUsedTargetColumn);
  formData.append("dataset", lastCleanedCsvBlob, "cleaned_data.csv");

  try {
    const response = await fetch("/generate_shap_plot", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(error);
    }
    button = document.getElementById("generate-shap-button");
    button.style.display = "none";
    document.getElementById('training-spinner-shap').style.display = "none";


    const result = await response.json();
    const shapImage = result.shap_summary_plot;
    shapPlot = shapImage;
    const plotCard = document.getElementById("shap-plots");
    plotCard.innerHTML = `
      <h4 data-i18n="shapSummary">${t("shapSummary")}</h4>
      <img src="data:image/png;base64,${shapImage}" alt="SHAP Summary" />
    `;
  } catch (error) {
    document.getElementById('training-spinner-shap').style.display = "none";
    console.error("Erreur SHAP:", error);
    showAlert(t("shapGenerationError") + ": " + error.message, "error");
  }
}


function downloadAllPlots() {
  const zip = new JSZip();
  const images = document.querySelectorAll('.plot-card img');

  images.forEach((img, index) => {
    const base64 = img.src.split(',')[1]; // Get only the base64 part
    const alt = img.alt.replace(/\s+/g, '_').toLowerCase(); // Safe filename
    zip.file(`${alt || 'plot_' + index}.png`, base64, { base64: true });
  });

  // Récupération du nom du dataset
  const fileInput = document.getElementById('upload-csv');
  const fileName = fileInput.files.length > 0 ? fileInput.files[0].name.replace(/\.csv$/, '') : 'dataset';

  // Génération d’un timestamp lisible
  const now = new Date();
  const timestamp = now.toISOString().replace(/[:\-T]/g, '_').split('.')[0]; // ex: 2025_05_30_14_45_12

  // Nom final du fichier zip
  const finalFileName = `${fileName}_plots_${timestamp}.zip`;

  zip.generateAsync({ type: "blob" })
    .then(function (content) {
      const link = document.createElement("a");
      link.href = URL.createObjectURL(content);
      link.download = finalFileName;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
}



document.getElementById('predict-csv').addEventListener('change', function () {
  const file = this.files[0];
  const acceptedExtensions = ['.csv', '.xls', '.xlsx', '.xlsm', '.arff'];
  if (file && acceptedExtensions.some(ext => file.name.endsWith(ext))) {
    this.style.display = 'none';
    document.getElementById('predict-csv-label').style.display = 'none';
    document.getElementById('predict-file-name').textContent = file.name;
    document.getElementById('predict-file-info').style.display = 'inline-block';

    // Affiche l'étape suivante
    document.getElementById('step-2-model').style.display = 'block';
  } else {
    showAlert(t("pleaseSelectCFile"), 'warning');
    this.value = '';
  }
});

function removePredictFile() {
  const input = document.getElementById('predict-csv');
  input.value = '';
  document.getElementById('predict-csv-label').style.display = 'inline-block';
  document.getElementById('predict-file-info').style.display = 'none';
  document.getElementById('predict-file-name').textContent = '';

  // Masquer les étapes suivantes
  document.getElementById('step-2-model').style.display = 'none';
  document.getElementById('step-3-predict').style.display = 'none';
  document.getElementById('prediction-results').style.display = 'none';
}

document.getElementById('predict-model-zip').addEventListener('change', function () {
  const file = this.files[0];
  if (file && file.name.endsWith('.zip')) {
    this.style.display = 'none';
    document.getElementById('predict-zip-label').style.display = 'none';
    document.getElementById('predict-model-name').textContent = file.name;
    document.getElementById('predict-model-info').style.display = 'inline-block';

    // Affiche le bouton prédire
    document.getElementById('step-3-predict').style.display = 'block';
  } else {
    showAlert(t("pleaseSelectZIP"), 'warning');
    this.value = '';
  }
});

function removePredictModel() {
  const input = document.getElementById('predict-model-zip');
  input.value = '';
  document.getElementById('predict-zip-label').style.display = 'inline-block';
  document.getElementById('predict-model-info').style.display = 'none';
  document.getElementById('predict-model-name').textContent = '';

  document.getElementById('step-3-predict').style.display = 'none';
}


function runPrediction() {
  const datasetInput = document.getElementById('predict-csv');
  const modelInput = document.getElementById('predict-model-zip');
  const dataset = datasetInput.files[0];
  const model = modelInput.files[0];

  if (!dataset || !model) {
    showAlert(t("selectDatasetAndModel"), 'warning');
    return;
  }

  const formData = new FormData();
  formData.append('dataset', dataset);
  formData.append('zip_model', model);

  fetch('/predict', {
    method: 'POST',
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        showAlert(t(data.error), 'error');
        return;
      }

      if (!data.preview || data.preview.length === 0) {
        showAlert("No prediction available.", 'warning');
        return;
      }

      const predictionResults = document.getElementById('prediction-results');
      predictionResults.innerHTML = `
        <h2 data-i18n="predictionResults">${t("predictionResults")}</h2>
        <div class="result-section">
          <h3 data-i18n="preview">${t("preview")}</h3>
          <div id="prediction-table-container"></div>
        </div>
      `;

      // TABLE DE PRÉVISUALISATION
      const tableContainer = document.getElementById('prediction-table-container');
      const table = document.createElement('table');
      table.id = 'prediction-table';
      table.className = 'preview-table';

      const header = Object.keys(data.preview[0]);
      const thead = document.createElement('thead');
      const headRow = document.createElement('tr');
      header.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col;
        headRow.appendChild(th);
      });
      thead.appendChild(headRow);
      table.appendChild(thead);

      const tbody = document.createElement('tbody');
      data.preview.forEach(row => {
        const tr = document.createElement('tr');
        header.forEach(col => {
          const td = document.createElement('td');
          td.textContent = row[col];
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      tableContainer.appendChild(table);

      // SECTION DES GRAPHIQUES
      if (data.plots && Object.keys(data.plots).length > 0) {
        let plotsHTML = `
          <div class="result-section">
            <h3 data-i18n="predictionPlots">${t("predictionPlots")}</h3>
        `;

        for (const [title, base64] of Object.entries(data.plots)) {
          const formattedTitle = title.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
          plotsHTML += `
            <div class="plot-card">
              <p><strong data-i18n="formattedTitle">${formattedTitle}</strong></p>
              <img src="data:image/png;base64,${base64}" alt="${formattedTitle}" />
            </div>
          `;
        }

        plotsHTML += `</div>`;
        predictionResults.innerHTML += plotsHTML;
      }

      // BOUTON DE TÉLÉCHARGEMENT DU CSV
      const downloadUrl = data.download_url || '#';
      predictionResults.innerHTML += `
        <div class="download-buttons-container">
          <a id="prediction-download-link" href="${downloadUrl}" data-i18n="downloadPredictions" class="download-button download-model" download>
            📥 ${t("downloadPredictions")}
          </a>
          <button class="download-button download-plots" data-i18n="downloadPredictionPlots" onclick="downloadAllPredictionPlots()">
            📊 ${t("downloadPredictionPlots")}
          </button>
        </div>
      `;

      predictionResults.style.display = 'block';
    })
    .catch(error => {
      showAlert(t("predictionError"), 'error');
    });
}

function downloadAllPredictionPlots() {
  const zip = new JSZip();
  const images = document.querySelectorAll('#prediction-results .plot-card img');

  images.forEach((img, index) => {
    const base64 = img.src.split(',')[1];
    const alt = img.alt.replace(/\s+/g, '_').toLowerCase();
    zip.file(`${alt || 'plot_' + index}.png`, base64, { base64: true });
  });

  const fileInput = document.getElementById('predict-csv');
  const fileName = fileInput.files.length > 0 ? fileInput.files[0].name.replace(/\.csv$/, '') : 'dataset';

  const now = new Date();
  const timestamp = now.toISOString().replace(/[:\-T]/g, '_').split('.')[0];
  const finalFileName = `${fileName}_prediction_plots_${timestamp}.zip`;

  zip.generateAsync({ type: "blob" }).then(function (content) {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(content);
    link.download = finalFileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });
}

async function sendChat() {
  const input = document.getElementById('chat-input');
  const sendButton = document.querySelector('#chat-footer button');
  const chatBox = document.getElementById('chat-box');
  const lang = localStorage.getItem("selectedLanguage") || "en";
  const message = input.value.trim();

  if (!message) return;

  input.disabled = true;
  sendButton.disabled = true;

  const userMsg = document.createElement('div');
  userMsg.className = 'chat-message user';
  userMsg.textContent = message;
  chatBox.appendChild(userMsg);
  chatBox.scrollTop = chatBox.scrollHeight;
  input.value = '';

  try {
    const payload = {
      message,
      lang
    };    
    if (Object.keys(summaryResults).length !== 0) {
      payload.summary = {
        text: summaryResults.summary, 
        feature_importance_plot: summaryResults.feature_importance_plot, 
        metrics_plot: summaryResults.metrics_plot 
      };
      if (shapPlot){
        payload.shap_summary_plot = shapPlot
      }
    }
    if (dataSetPreview) {
      payload.markdown_preview = dataSetPreview;
    }
    const response = await fetch('/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const errorData = await response.json();
      showAlert(t(errorData.error || 'error_unknown'), 'error');
      return;
    }

    const data = await response.json();
    const aiReply = data.response;

    const aiMsg = document.createElement('div');
    aiMsg.className = 'chat-message ai';
    aiMsg.innerHTML = marked.parse(aiReply);
    chatBox.appendChild(aiMsg);
    chatBox.scrollTop = chatBox.scrollHeight;

  } catch (error) {
    const errorMsg = document.createElement('div');
    errorMsg.className = 'chat-message ai';
    errorMsg.textContent = t('error_chat', 'error');
    chatBox.appendChild(errorMsg);
    chatBox.scrollTop = chatBox.scrollHeight;
  } finally {
    input.disabled = false;
    sendButton.disabled = false;
    input.focus();
  }
}



document.getElementById('chat-input').addEventListener('keydown', function (event) {
  // Si on appuie sur Entrée sans Maj
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault(); // empêche le saut de ligne
    sendChat(); 
  }
});

function toggleChatSidebar() {
  const sidebar = document.getElementById('chat-sidebar');
  const toggleButton = document.getElementById('toggle-chat-btn');
  const wrapper = document.querySelector('.page-wrapper');

  const isCollapsed = sidebar.classList.toggle('collapsed');

  if (isCollapsed) {
    toggleButton.classList.remove('hidden');
    wrapper.classList.remove('chat-open'); 
  } else {
    toggleButton.classList.add('hidden');
    wrapper.classList.add('chat-open'); 
  }
}


function showAlert(message, type = 'error', duration = 10000) {
  const alertBox = document.getElementById('custom-alert');
  alertBox.textContent = message;

  alertBox.className = 'custom-alert'; // reset
  if (type === 'success') alertBox.classList.add('success');
  else if (type === 'warning') alertBox.classList.add('warning');
  else alertBox.classList.add('error');

  alertBox.classList.remove('hidden');

  setTimeout(() => {
    alertBox.classList.add('hidden');
  }, duration);
}
