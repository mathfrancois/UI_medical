from flask import Flask, request, render_template, jsonify, send_file
import os
from model_utils import train_model, predict_model
from datetime import datetime
import shutil
import zipfile
from werkzeug.utils import secure_filename
import pandas as pd

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
DECOMPRESSION_FOLDER = 'decompression'
PREDICTION_FOLDER = 'prediction'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

def create_training_folder(dataset_filename, base_dir=MODEL_FOLDER):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataset_name = os.path.splitext(os.path.basename(dataset_filename))[0]
    folder_name = f"{dataset_name}_{timestamp}"
    zip_filename = f"model_{dataset_name}_{timestamp}.zip"
    path = os.path.join(base_dir, folder_name)
    os.makedirs(path, exist_ok=True)
    return path, folder_name, zip_filename

@app.route('/')
def index():
    return render_template('interface.html')

@app.route('/train', methods=['POST'])
def train():
    try:
        file = request.files['csv_file']
        target_column = request.form['target_column'].strip()
        time_limit = int(request.form.get('time_limit', 60))

        if time_limit < 10 or time_limit > 600:
            return jsonify({"error": "Time limit must be between 10 and 600 seconds."}), 400

        csv_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(csv_path)

        df = pd.read_csv(csv_path)

        if df[target_column].isnull().all():
            return jsonify({"error": f"Target column '{target_column}' contains only missing values."}), 400

        training_folder, training_id, zip_filename = create_training_folder(file.filename)

        include_shap = request.form.get("include_shap", "false").lower() == "true"

        metrics = train_model(csv_path, target_column, time_limit, training_folder, include_shap)

        # Créer le fichier zip avec le bon nom
        zip_path = os.path.join(MODEL_FOLDER, zip_filename)
        shutil.make_archive(base_name=zip_path.replace('.zip', ''), format='zip', root_dir=training_folder)

        metrics['download_url'] = f"/download_model/{zip_filename}"
        return jsonify(metrics)

    except Exception as e:
        return jsonify({"error": f"Training failed: {str(e)}"}), 500

@app.route('/download_model/<zip_filename>')
def download_model(zip_filename):
    zip_path = os.path.join(MODEL_FOLDER, secure_filename(zip_filename))
    if not os.path.exists(zip_path):
        return "File not found", 404
    return send_file(zip_path, as_attachment=True)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        dataset_file = request.files.get('csv_dataset')
        model_zip = request.files.get('zip_model')

        if not dataset_file or not model_zip:
            return jsonify({"error": "CSV and/or model file missing."}), 400

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dataset_name = os.path.splitext(secure_filename(dataset_file.filename))[0]
        identifier = f"{timestamp}_{dataset_name}"

        csv_filename = f"{identifier}.csv"
        csv_path = os.path.join(UPLOAD_FOLDER, csv_filename)
        dataset_file.save(csv_path)

        decompression_folder = os.path.join(DECOMPRESSION_FOLDER, f"predict_{identifier}")
        os.makedirs(decompression_folder, exist_ok=True)

        zip_path = os.path.join(decompression_folder, "model.zip")
        model_zip.save(zip_path)

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(decompression_folder)
        except zipfile.BadZipFile:
            return jsonify({"error": "Uploaded model file is not a valid ZIP archive."}), 400

        result = predict_model(csv_path, decompression_folder)
        predictions = result['predictions']  
        plots = result['plots']             

        pred_filename = f"{identifier}_predictions.csv"
        pred_path = os.path.join(PREDICTION_FOLDER, pred_filename)
        pd.DataFrame(predictions).to_csv(pred_path, index=False)

        return jsonify({
            "preview": predictions[:5],
            "download_url": f"/download_prediction/{pred_filename}",
            "plots": plots
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500



@app.route('/download_prediction/<filename>')
def download_prediction(filename):
    filepath = os.path.join(PREDICTION_FOLDER, filename)
    if not os.path.exists(filepath):
        return "File not found", 404
    return send_file(filepath, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
