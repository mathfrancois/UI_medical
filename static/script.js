let parsedData = [];

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
  if (file && file.name.endsWith('.csv')) {
    // Cacher l'input et le bouton
    this.style.display = 'none';
    document.getElementById('upload-csv-label').style.display = 'none';

    document.getElementById('csv-file-name').textContent = file.name;
    document.getElementById('csv-file-info').style.display = 'inline';

    // Aperçu (inchangé)
    document.getElementById('csv-preview').style.display = 'block';
    const reader = new FileReader();
    reader.onload = function (e) {
      const previewTable = document.getElementById('preview-table');
      const content = e.target.result;
      const parsed = Papa.parse(content, {
        header: false,
        skipEmptyLines: true
      });
      const rows = parsed.data;

      const header = rows[0];
      const numColumns = header.length;
      const numRows = rows.length - 1;

      let nanCount = 0;

      for (let i = 1; i < rows.length; i++) {
        nanCount += rows[i].filter(cell =>
          cell.trim() === "" || cell.trim().toLowerCase() === "nan"
        ).length;
      }


      // Afficher les stats
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

      const targetSelect = document.getElementById("target-column");
      targetSelect.innerHTML = "";
      header.forEach(col => {
        const option = document.createElement("option");
        option.value = col;
        option.textContent = col;
        targetSelect.appendChild(option);
      });
    };
    reader.readAsText(file);
  } else {
    showAlert("Please select a .csv file", 'warning');
    this.value = '';
  }
});


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

}

function startTraining() {
  const fileInput = document.getElementById('upload-csv');
  const file = fileInput.files[0];
  const targetColumn = document.getElementById('target-column').value;

  if (!file || !targetColumn) {
    showAlert("Please select a CSV file and a target column.", 'warning');
    return;
  }

  document.getElementById('training-spinner').style.display = "inline-block";

  const formData = new FormData();
  formData.append('csv_file', file);
  formData.append('target_column', targetColumn);
  formData.append('time_limit', timeLimitSeconds);

  const includeShap = document.getElementById("shap-toggle").checked;
  formData.append("include_shap", includeShap);

  fetch('/train', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    if (data.error) {
      showAlert(data.error, 'error');
      return;
    }

    document.getElementById('training-spinner').style.display = "none";
    const resultsDiv = document.getElementById('training-results');

    // Début structure principale
    resultsDiv.innerHTML = `<h2>Training Results</h2>`;

    // -------- Training Summary Section --------
    let summaryHTML = `
      <div class="result-section" id="training-summary-section">
        <h3>Training Summary</h3>

        <div class="subsection">
          <h4>General Information</h4>
          <p><strong>Detected Task:</strong> ${data.task_type}</p>
          <p><strong>Selected Model:</strong> ${data.best_model}</p>
          <p><strong>Training Time:</strong> ${data.train_time.toFixed(2)} seconds</p>
        </div>
    `;

    const metrics = data.metrics;
    let metricsHTML = '';
    let plotsHTML = `
      <div class="subsection">
        <h4>Plots</h4>
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
          <thead><tr><th>Metric</th><th>Value</th></tr></thead>
          <tbody>${metricsHTML}</tbody>
        </table>
      </div>
    `;

    plotsHTML += `</div>`; // Fin plots
    summaryHTML += plotsHTML + `</div>`; // Fin Training Summary
    resultsDiv.innerHTML += summaryHTML;

    // -------- Model Explainability Section --------
    let explainabilityHTML = `
      <div class="result-section" id="model-explainability-section">
        <h3>Model Explainability</h3>
    `;

    if (data.feature_importance_plot) {
      explainabilityHTML += `
        <div class="plot-card">
          <h4>Feature Importance</h4>
          <img src="data:image/png;base64,${data.feature_importance_plot}" alt="Feature Importance Plot" />
        </div>
      `;
    }

    if (data.shap_plots) {
      explainabilityHTML += `
        <div class="plot-card">
          <h4>SHAP Summary Plot</h4>
          <img src="data:image/png;base64,${data.shap_plots}" alt="SHAP Summary" />
        </div>
      `;
    }

    explainabilityHTML += `</div>`; // Fin Model Explainability
    resultsDiv.innerHTML += explainabilityHTML;

    // -------- Download Buttons --------
    resultsDiv.innerHTML += `
      <div class="download-buttons-container">
        <a id="download-link" href="#" class="download-button download-model" download>
          🧠 Download Trained Model (.zip)
        </a>
        <button class="download-button download-plots" onclick="downloadAllPlots()">
          📊 Download All Plots (.zip)
        </button>
      </div>
    `;

    if (data.download_url) {
      document.getElementById('download-link').href = data.download_url;
    }

    resultsDiv.style.display = 'block';
  })

  .catch(error => {
    document.getElementById('training-spinner').style.display = "none";
    console.error("Error during training:", error);
    showAlert("An error occurred during training.", 'error');
  });
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
  if (file && file.name.endsWith('.csv')) {
    this.style.display = 'none';
    document.getElementById('predict-csv-label').style.display = 'none';
    document.getElementById('predict-file-name').textContent = file.name;
    document.getElementById('predict-file-info').style.display = 'inline-block';

    // Affiche l'étape suivante
    document.getElementById('step-2-model').style.display = 'block';
  } else {
    showAlert("Please select a CSV file", 'warning');
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
    showAlert("Please select a ZIP file", 'warning');
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
    showAlert("Please select a dataset (.csv) and a model (.zip)", 'warning');
    return;
  }

  const formData = new FormData();
  formData.append('csv_dataset', dataset);
  formData.append('zip_model', model);

  fetch('/predict', {
    method: 'POST',
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        showAlert(data.error, 'error');
        return;
      }

      if (!data.preview || data.preview.length === 0) {
        showAlert("No prediction available.", 'warning');
        return;
      }

      const predictionResults = document.getElementById('prediction-results');
      predictionResults.innerHTML = `
        <h2>Prediction Results</h2>
        <div class="result-section">
          <h3>Preview</h3>
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
            <h3>Prediction Plots</h3>
        `;

        for (const [title, base64] of Object.entries(data.plots)) {
          const formattedTitle = title.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
          plotsHTML += `
            <div class="plot-card">
              <p><strong>${formattedTitle}</strong></p>
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
          <a id="prediction-download-link" href="${downloadUrl}" class="download-button download-model" download>
            📥 Download predictions (.csv)
          </a>
          <button class="download-button download-plots" onclick="downloadAllPredictionPlots()">
            📊 Download Plots (.zip)
          </button>
        </div>
      `;

      predictionResults.style.display = 'block';
    })
    .catch(error => {
      console.error(error);
      showAlert("An error has occurred during prediction.", 'error');
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





function sendChat() {
  const input = document.getElementById('chat-input');
  const message = input.value;
  if (!message) return;
  const box = document.getElementById('chat-box');
  const p = document.createElement('p');
  p.textContent = message;
  box.appendChild(p);
  input.value = "";

  // Simuler une réponse
  const reply = document.createElement('p');
  reply.textContent = "Tip: You can choose any column to predict. The model will learn from the others.";
  reply.style.fontStyle = "italic";
  box.appendChild(reply);
  box.scrollTop = box.scrollHeight;
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
