from flask import Flask, request, render_template, jsonify, send_file
import os
from model_utils import train_model, predict_model, generate_shap_plot, UserError
from datetime import datetime
import shutil
import zipfile
from werkzeug.utils import secure_filename
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
DECOMPRESSION_FOLDER = 'decompression'
PREDICTION_FOLDER = 'prediction'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # ou fixe-la directement (non recommandé en clair)

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
        file = request.files['file']
        target_column = request.form['target_column'].strip()
        time_limit = int(request.form.get('time_limit', 60))

        if time_limit < 10 or time_limit > 600:
            raise UserError("error_time_limit_range")

        path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(path)
        ext = os.path.splitext(path)[1].lower()

        if ext == '.csv':
            df = pd.read_csv(path)
        elif ext in ['.xls', '.xlsx', '.xlsm']:
            df = pd.read_excel(path)
        elif ext == '.arff':
            from scipy.io import arff
            data, meta = arff.loadarff(path)
            df = pd.DataFrame(data)
            for col in df.select_dtypes([object]).columns:
                df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        else:
            raise UserError("error_unsupported_file_format")

        if target_column not in df.columns:
            raise UserError("error_target_not_in_columns")

        if df[target_column].isnull().all():
            raise UserError("error_target_all_missing")

        training_folder, training_id, zip_filename = create_training_folder(file.filename)
        include_shap = request.form.get("include_shap", "false").lower() == "true"

        metrics = train_model(df, target_column, time_limit, training_folder, include_shap)

        zip_path = os.path.join(MODEL_FOLDER, zip_filename)
        shutil.make_archive(base_name=zip_path.replace('.zip', ''), format='zip', root_dir=training_folder)

        metrics['download_url'] = f"/download_model/{zip_filename}"
        return jsonify(metrics)

    except UserError as ue:
        return jsonify({"error": ue.message_key}), ue.status_code
    except Exception as e:
        return jsonify({"error": "error_training_failed"}), 500

@app.route('/download_model/<zip_filename>')
def download_model(zip_filename):
    zip_path = os.path.join(MODEL_FOLDER, secure_filename(zip_filename))
    if not os.path.exists(zip_path):
        return "File not found", 404
    return send_file(zip_path, as_attachment=True)

@app.route('/generate_shap_plot', methods=['POST'])
def shap_plot():
    try:
        if 'dataset' not in request.files:
            raise UserError("error_no_dataset")

        file = request.files['dataset']
        df = pd.read_csv(file)

        model_path = request.form.get('model_path')
        target_column = request.form.get('target_column')

        if not model_path or not os.path.exists(model_path):
            raise UserError("error_invalid_model_path")

        if not target_column:
            raise UserError("error_missing_target_column")

        result = generate_shap_plot(model_path, df, target_column)
        return jsonify(result)

    except UserError as ue:
        return jsonify({"error": ue.message_key}), ue.status_code
    except Exception as e:
        return jsonify({"error": "error_shap_failed"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        dataset_file = request.files.get('dataset')
        model_zip = request.files.get('zip_model')

        if not dataset_file or not model_zip:
            raise UserError("error_missing_dataset_or_model")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dataset_name = os.path.splitext(secure_filename(dataset_file.filename))[0]
        identifier = f"{timestamp}_{dataset_name}"

        file_ext = os.path.splitext(dataset_file.filename)[1].lower()
        if file_ext not in ['.csv', '.xls', '.xlsx', '.xlsm', '.arff']:
            raise UserError("error_unsupported_file_format")

        original_filename = f"{identifier}{file_ext}"
        file_path = os.path.join(UPLOAD_FOLDER, original_filename)
        dataset_file.save(file_path)

        decompression_folder = os.path.join(DECOMPRESSION_FOLDER, f"predict_{identifier}")
        os.makedirs(decompression_folder, exist_ok=True)

        zip_path = os.path.join(decompression_folder, "model.zip")
        model_zip.save(zip_path)

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(decompression_folder)
        except zipfile.BadZipFile:
            raise UserError("error_invalid_zip")

        result = predict_model(file_path, decompression_folder)
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

    except UserError as ue:
        return jsonify({"error": ue.message_key}), ue.status_code
    except Exception as e:
        return jsonify({"error": "error_prediction_failed"}), 500

@app.route('/download_prediction/<filename>')
def download_prediction(filename):
    filepath = os.path.join(PREDICTION_FOLDER, filename)
    if not os.path.exists(filepath):
        return "File not found", 404
    return send_file(filepath, as_attachment=True)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message', '')
        summary = request.json.get('summary', None)
        lang = request.json.get('lang', 'en')

        if not user_input:
            return jsonify({"error": "error_empty_message"}), 400
        
        lang_prompts = {
            "en": "Please respond in English.",
            "fr": "Merci de répondre en français.",
            "es": "Por favor responde en español."
        }

        lang_prompt = lang_prompts.get(lang, lang_prompts["en"])

        system_prompt = f"""
        {lang_prompt}
        Tu es un assistant virtuel spécialisé en autoML, dédié à des professionnels médicaux non experts en machine learning.
        Tu aides à comprendre comment utiliser une interface qui permet d'entraîner des modèles sur des données médicales, choisir la colonne cible,
        paramétrer le temps d'entraînement, supprimer des colonnes du dataset avant entraînement, visualiser les résultats (métriques, matrices de confusion, courbes ROC, graphiques SHAP),
        et prédire sur de nouvelles données.
        Tu expliques les concepts simplement, sans jargon technique, et guides l'utilisateur sur comment améliorer son dataset, comprendre ses modèles et interpréter les résultats.

        Exemples de questions :
        - Qu’est-ce qu’une matrice de confusion et comment l’interpréter ?
        - Que signifie le graphique SHAP ?
        - Comment puis-je améliorer mon dataset pour avoir un meilleur modèle ?
        - Que faire si mon modèle a un score de précision faible ?
        """

        messages = [{"role": "system", "content": system_prompt}]

        if summary:
            messages.append({
                "role": "user",
                "content": f"Voici le résumé des résultats de l'entraînement :\n{summary}"
            })

        messages.append({"role": "user", "content": user_input})

        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=0.7,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )

        response = completion.choices[0].message.content
        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": "error_chat"}), 500



if __name__ == '__main__':
    app.run(debug=True)
