let parsedData = [];

function updateTimeDisplay() {
    const slider = document.getElementById("training-time-limit");
    const display = document.getElementById("time-limit-display");
    display.textContent = slider.value;
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
      const content = e.target.result;
      const rows = content.split("\n").filter(r => r.trim() !== "");
      const previewTable = document.getElementById('preview-table');
      previewTable.innerHTML = "";

      const header = rows[0].split(",");
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
        const row = rows[i].split(",");
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
    alert("Please select a .csv file");
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
    alert("Please select a CSV file and a target column.");
    return;
  }

  document.getElementById('training-spinner').style.display = "inline-block";

  const formData = new FormData();
  formData.append('csv_file', file);
  formData.append('target_column', targetColumn);
  formData.append('time_limit', timeLimitSeconds);

  fetch('/train', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    if (data.error) {
      alert(data.error);
      return;
    }
    document.getElementById('training-spinner').style.display = "none";
    const resultsDiv = document.getElementById('training-results');
    resultsDiv.innerHTML = `
      <h2>Training Results</h2>
      <div class="result-section">
        <h3>General Information</h3>
        <p><strong>Detected Task:</strong> ${data.task_type}</p>
        <p><strong>Selected Model:</strong> ${data.best_model}</p>
        <p><strong>Training Time:</strong> ${data.train_time.toFixed(2)} seconds</p>
      </div>
    `;

    const metrics = data.metrics;

    if (data.task_type == "regression") {
      let metricsHTML = `
        <div class="result-section">
          <h3>Regression Metrics</h3>
          <table class="metrics-table">
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>
      `;

      let plotsHTML = `
        <div class="result-section">
          <h3>Plots</h3>
      `;

      for (const metric in metrics) {
        const value = metrics[metric].value;
        const plotBase64 = metrics[metric].plot;
        const plotHist = metrics[metric].plot_hist;
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

      metricsHTML += `</tbody></table></div>`;
      plotsHTML += `</div>`;

      resultsDiv.innerHTML += metricsHTML + plotsHTML;
    }

    else { // Classification or other
      let metricsHTML = `
        <div class="result-section">
          <h3>Classification Metrics</h3>
          <table class="metrics-table">
            <thead><tr><th>Metric</th><th>Value</th></tr></thead>
            <tbody>
      `;

      let plotsHTML = `
        <div class="result-section">
          <h3>Plots</h3>
      `;

      for (const metric in metrics) {
        const value = metrics[metric].value;
        const plotBase64 = metrics[metric].plot;
        const metricLabel = metric.toUpperCase();
        const formattedValue = value.toFixed(4);

        metricsHTML += `<tr><td>${metricLabel}</td><td>${formattedValue}</td></tr>`;

        if (plotBase64) {
          plotsHTML += `
            <div class="plot-card">
              <p><strong>${metricLabel}</strong></p>
              <img src="data:image/png;base64,${plotBase64}" alt="${metricLabel} Plot" />
            </div>
          `;
        }
      }

      metricsHTML += `</tbody></table></div>`;
      plotsHTML += `</div>`;

      resultsDiv.innerHTML += metricsHTML + plotsHTML;
    }

    resultsDiv.innerHTML += `
      <a id="download-link" href="#" class="download-button" download>
        📦 Download Model (.zip)
      </a>`;

    if (data.download_url) {
      document.getElementById('download-link').href = data.download_url;
    }

    resultsDiv.style.display = 'block';
  })
  .catch(error => {
    document.getElementById('training-spinner').style.display = "none";
    console.error("Error during training:", error);
    alert("An error occurred during training.");
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
    alert("Please select a CSV file");
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
    alert("Please select a ZIP file");
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
  const modelInput = document.getElementById('predict-model-zip')
  const dataset = datasetInput.files[0];
  const model = modelInput.files[0]

  if (!dataset || !model) {
    alert("Please select a dataset (.csv) and a model (.zip)");
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
        alert(data.error);
        return;
      }
      if (!data.preview || data.preview.length === 0) {
        alert("No prediction available.");
        return;
      }

      const table = document.getElementById('prediction-table');
      table.innerHTML = '';

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

      document.getElementById('prediction-results').style.display = 'block';
      const predictionResults = document.getElementById('prediction-results');

      const oldDownloadLink = document.getElementById('prediction-download-link');

      const predictionDownloadLink = document.createElement('a');
      predictionDownloadLink.id = 'prediction-download-link';
      predictionDownloadLink.href = data.download_url; 
      predictionDownloadLink.className = 'download-button';
      predictionDownloadLink.download = 'predictions.csv';
      predictionDownloadLink.textContent = '📥 Download predictions';

      predictionResults.appendChild(predictionDownloadLink);

      predictionResults.style.display = 'block';
    })
    .catch(error => {
      console.error("Prediction error :", error);
      alert("An error has occurred during prediction.");
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
