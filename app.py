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
import logging
from logging.handlers import RotatingFileHandler
import threading

training_threads = {}
stop_events = {}

training_results = {}

load_dotenv()

log_formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')

os.makedirs("logs", exist_ok=True)

log_file = "logs/server.log"
file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)

error_file = "logs/errors.log"
error_handler = RotatingFileHandler(error_file, maxBytes=5 * 1024 * 1024, backupCount=3)
error_handler.setFormatter(log_formatter)
error_handler.setLevel(logging.WARNING)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(logging.DEBUG)

logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG", "False").lower() == "true" else logging.INFO,
    handlers=[file_handler, error_handler, stream_handler]
)


logger = logging.getLogger(__name__)
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
DECOMPRESSION_FOLDER = 'decompression'
PREDICTION_FOLDER = 'prediction'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(PREDICTION_FOLDER, exist_ok=True)

app.config['ENV'] = os.getenv("FLASK_ENV", "production")
app.config['DEBUG'] = os.getenv("DEBUG", "False").lower() == "true"
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx', 'xlsm', 'arff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preview_dataset(df, max_rows=5):
    try:
        return df.head(max_rows).to_markdown(index=False)
    except Exception as e:
        logger.warning(f"Erreur lors de l'aperçu du dataset : {e}")
        return f"Erreur lors de l'aperçu du dataset : {e}"

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
    logger.info("Affichage de la page d'accueil.")
    return render_template('interface.html')

@app.route('/train', methods=['POST'])
def train():
    try:
        logger.info("Début de l'entraînement du modèle.")
        file = request.files.get('file')
        if not file or file.filename == '':
            raise UserError("error_no_file")
        if not allowed_file(file.filename):
            raise UserError("error_unsupported_file_format")

        target_column = request.form['target_column'].strip()
        time_limit = int(request.form.get('time_limit', 60))
        if time_limit < 10 or time_limit > 600:
            raise UserError("error_time_limit_range")

        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)
        logger.debug(f"Fichier reçu : {filename}")

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

        markdown_preview = preview_dataset(df)
        training_folder, training_id, zip_filename = create_training_folder(filename)
        logger.info(f"Dossier d'entraînement créé : {training_id}")

        stop_event = threading.Event()
        stop_events[training_id] = stop_event

        def training_task():
            try:
                metrics = train_model(df, target_column, time_limit, training_folder, stop_event)
                zip_path = os.path.join(MODEL_FOLDER, zip_filename)
                shutil.make_archive(base_name=zip_path.replace('.zip', ''), format='zip', root_dir=training_folder)
                logger.info(f"Modèle compressé et sauvegardé : {zip_filename}")
                metrics['download_url'] = f"/download_model/{zip_filename}"
                metrics['markdown_preview'] = markdown_preview
                metrics['training_id'] = training_id

                training_results[training_id] = metrics
                logger.info(f"Entraînement terminé avec succès pour {training_id}.")

                training_threads.pop(training_id, None)
                stop_events.pop(training_id, None)
            except Exception as e:
                logger.error(f"Erreur dans le thread d'entraînement : {e}")
                training_threads.pop(training_id, None)
                stop_events.pop(training_id, None)

        training_thread = threading.Thread(target=training_task, daemon=True)
        training_threads[training_id] = training_thread
        training_thread.start()

        return jsonify({"training_id": training_id})

    except UserError as ue:
        logger.warning(f"Erreur utilisateur pendant l'entraînement : {ue.message_key}")
        return jsonify({"error": ue.message_key}), ue.status_code
    except Exception as e:
        logger.error(f"Erreur critique pendant l'entraînement : {e}")
        return jsonify({"error": "error_training_failed"}), 500
    
@app.route('/training_result/<training_id>', methods=['GET'])
def get_training_result(training_id):
    if training_id in training_results:
        result = training_results.pop(training_id)  
        return jsonify(result)
    else:
        return '', 202  

    
@app.route('/stop_training/<training_id>', methods=['POST'])
def stop_training(training_id):
    stop_event = stop_events.get(training_id)
    if stop_event:
        stop_event.set()
        return jsonify({"message": "training_stopped"})
    else:
        return jsonify({"error": "training_id_not_found"}), 404


@app.route('/download_model/<zip_filename>')
def download_model(zip_filename):
    zip_path = os.path.join(MODEL_FOLDER, secure_filename(zip_filename))
    if not os.path.exists(zip_path):
        logger.warning(f"Téléchargement échoué : fichier introuvable {zip_filename}")
        return "File not found", 404
    logger.info(f"Téléchargement du modèle : {zip_filename}")
    return send_file(zip_path, as_attachment=True)
    

@app.route('/generate_shap_plot', methods=['POST'])
def shap_plot():
    try:
        logger.info("Génération du graphe SHAP.")
        file = request.files.get('dataset')
        if not file or not allowed_file(file.filename):
            raise UserError("error_no_dataset")

        df = pd.read_csv(file)
        model_path = request.form.get('model_path')
        target_column = request.form.get('target_column')

        if not model_path or not os.path.exists(model_path):
            raise UserError("error_invalid_model_path")
        if not target_column:
            raise UserError("error_missing_target_column")

        result = generate_shap_plot(model_path, df, target_column)
        logger.info("Graphe SHAP généré avec succès.")
        return jsonify(result)

    except UserError as ue:
        logger.warning(f"Erreur utilisateur SHAP : {ue.message_key}")
        return jsonify({"error": ue.message_key}), ue.status_code
    except Exception as e:
        logger.error(f"Erreur critique SHAP : {e}")
        return jsonify({"error": "error_shap_failed"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Début de la prédiction.")
        dataset_file = request.files.get('dataset')
        model_zip = request.files.get('zip_model')

        if not dataset_file or not model_zip:
            raise UserError("error_missing_dataset_or_model")
        if not allowed_file(dataset_file.filename):
            raise UserError("error_unsupported_file_format")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dataset_name = os.path.splitext(secure_filename(dataset_file.filename))[0]
        identifier = f"{timestamp}_{dataset_name}"

        file_ext = os.path.splitext(dataset_file.filename)[1].lower()
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

        logger.info(f"Prédictions générées avec succès : {pred_filename}")
        return jsonify({
            "preview": predictions[:5],
            "download_url": f"/download_prediction/{pred_filename}",
            "plots": plots
        })

    except UserError as ue:
        logger.warning(f"Erreur utilisateur prédiction : {ue.message_key}")
        return jsonify({"error": ue.message_key}), ue.status_code
    except Exception as e:
        logger.error(f"Erreur critique prédiction : {e}")
        return jsonify({"error": "error_prediction_failed"}), 500

@app.route('/download_prediction/<filename>')
def download_prediction(filename):
    filepath = os.path.join(PREDICTION_FOLDER, secure_filename(filename))
    if not os.path.exists(filepath):
        logger.warning(f"Fichier de prédiction non trouvé : {filename}")
        return "File not found", 404
    logger.info(f"Téléchargement des prédictions : {filename}")
    return send_file(filepath, as_attachment=True)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message', '')
        summary = request.json.get('summary', None)
        lang = request.json.get('lang', 'en')
        dataset = request.json.get('markdown_preview', None)
        stats = request.json.get('stats', None)
        shap_plot_base64 = request.json.get('shap_summary_plot', None)
        plot_predictions = request.json.get('plots_prediction_results', None)

        if not user_input:
            return jsonify({"error": "error_empty_message"}), 400

        logger.info(f"Requête utilisateur au chatbot reçue. Langue : {lang}")

        lang_prompts = {
            "en": "Please respond in English.",
            "fr": "Merci de répondre en français.",
            "es": "Por favor responde en español."
        }

        lang_prompt = lang_prompts.get(lang, lang_prompts["en"])
        print(f"Langue sélectionnée : {lang}")

        system_prompt = f"""
        {lang_prompt}
        Ne souligne pas le fait que tu répond dans une langue spécifique.
        Tu es un assistant virtuel spécialisé en autoML, dédié à des professionnels médicaux non experts en machine learning.
        Tu aides à comprendre comment utiliser une interface qui permet d'entraîner des modèles sur des données médicales, choisir la colonne cible,
        paramétrer le temps d'entraînement, supprimer des colonnes du dataset avant entraînement, visualiser les résultats (métriques, matrices de confusion, courbes ROC, graphiques SHAP),
        et l'utilisateur peut ensuite télécharger le modèle entraîné s'il lui va et télécharger les images des graphiques des résultats. Ensuite, l'utilisateur
        peut aller dans la section de prédiction, télécharger le modèle entraîné et entrer son dataset de prédiction pour obtenir des prédictions.
        Tu expliques les concepts simplement, sans jargon technique, et guides l'utilisateur sur comment améliorer son dataset, comprendre ses modèles et interpréter les résultats.
        Il faut savoir que le modèle est généré grâce à AutoGluon en AutoML, donc l'utilisateur n'a pas la main pour choisir le modèle et les hyperparamètres.
        De plus, pour l'instant, l'utilisateur ne peut que enlever des colonnes du dataset avant entrainement depuis l'interface pour influencer le modèle produit par AutoGluon pour l'instant.

        Exemples de questions :
        - Qu’est-ce qu’une matrice de confusion et comment l’interpréter ?
        - Que signifie le graphique SHAP ?
        - Comment puis-je améliorer mon dataset pour avoir un meilleur modèle ?
        - Que faire si mon modèle a un score de précision faible ?
        """

        messages = [{"role": "system", "content": system_prompt}]
        KEYWORDS_BY_LANG = {
            "fr": ["données", "dataset", "colonnes", "variables", "analyse", "statistiques", "modèle", "entraînement", "target"],
            "en": ["data", "dataset", "columns", "features", "variables", "analysis", "statistics", "model", "training", "target"],
            "es": ["datos", "dataset", "conjunto de datos", "columnas", "variables", "análisis", "estadísticas", "modelo", "entrenamiento", "objetivo"],
        }
        keywords = KEYWORDS_BY_LANG.get(lang, [])

        # Si aucune donnée n'a été upload et le user pose une question dessus
        if dataset is None and any(kw in user_input.lower() for kw in keywords):
            messages.append({
                "role": "assistant",
                "content": "Je ne vois pas encore de dataset ou de résultats d'entraînement. Pour que je puisse vous aider, vous devez d’abord lancer un entraînement via l’interface avec vos données."
        })

        if dataset:
            messages.append({"role": "user", "content": f"Aperçu du dataset fourni :\n{dataset}"})
        
        if stats:
            messages.append({"role": "user", "content": f"Statistiques du dataset :\n{stats}"})

        if summary:
            messages.append({"role": "user", "content": f"Résumé des résultats d'entraînement :\n{summary}"})
            if (plot := summary.get("feature_importance_plot")):
                messages.append({"role": "user", "content": [{"type": "text", "text": "Voici le graphe de l'importance des variables."}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{plot}"}}]})
            for metric_name, metric_obj in summary.get("metrics_plot", {}).items():
                base64_plot = metric_obj.get("plot")
                if base64_plot:
                    messages.append({"role": "user", "content": [{"type": "text", "text": f"Voici le graphe {metric_name}."}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_plot}"}}]})

        if shap_plot_base64:
            messages.append({"role": "user", "content": [{"type": "text", "text": "Voici le graphe SHAP de l'entraînement."}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{shap_plot_base64}"}}]})

        if plot_predictions:
            for name_plot, plot in plot_predictions.items():
                if plot:
                    messages.append({"role": "user", "content": [{"type": "text", "text": f"Voici le graphe {name_plot} des résultats de prédiction."}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{plot}"}}]})

        messages.append({"role": "user", "content": user_input})

        client = Groq(api_key=GROQ_API_KEY)
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=0.7,
            max_completion_tokens=1024,
            top_p=1,
            stream=False
        )

        logger.info("Réponse générée par le chatbot avec succès.")
        return jsonify({"response": completion.choices[0].message.content})

    except Exception as e:
        logger.error(f"Erreur chatbot : {e}")
        return jsonify({"error": "error_chat"}), 500

if __name__ == '__main__':
    logger.info("Lancement du serveur Flask.")
    app.run(debug=app.config['DEBUG'], env=app.config['ENV'])
