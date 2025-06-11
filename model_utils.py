from autogluon.tabular import TabularPredictor, TabularDataset
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  

import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import numpy as np
import shap


from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

class UserError(Exception):
    def __init__(self, message_key, status_code=400):
        super().__init__(message_key)
        self.message_key = message_key
        self.status_code = status_code

def summarize_feature_importance(importances, top_k=5):
    importances_sorted = importances.sort_values('importance', ascending=False)
    top_features = importances_sorted.head(top_k)
    summary = "Top {} variables les plus importantes selon le modèle :\n".format(top_k)
    for feature, row in top_features.iterrows():
        summary += f"- {feature} avec une importance relative de {row['importance']:.3f}\n"
    return summary

def summarize_confusion_matrix(y_true, y_pred, class_labels):
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    total = cm.sum()
    correct = np.trace(cm)
    accuracy = correct / total
    summary = f"La matrice de confusion montre une précision globale de {accuracy:.2%}.\n"
    for i, label in enumerate(class_labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        summary += (f"Pour la classe '{label}':\n"
                    f"  - Vrais positifs: {tp}\n"
                    f"  - Faux négatifs: {fn}\n"
                    f"  - Faux positifs: {fp}\n")
    return summary

def summarize_roc_auc(y_true, y_proba, pos_label):
    auc_score = roc_auc_score(y_true, y_proba)
    interpret = "excellent" if auc_score > 0.9 else "bon" if auc_score > 0.75 else "modéré"
    return f"La courbe ROC a un score AUC de {auc_score:.3f}, ce qui est {interpret} pour distinguer les classes."

def summarize_f1_per_class(y_true, y_pred, class_labels):
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=class_labels)
    summary = "F1 score par classe :\n"
    for label, score in zip(class_labels, f1_per_class):
        summary += f"- Classe '{label}': {score:.3f}\n"
    return summary

def summarize_regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    errors = y_pred - y_true
    mean_error = errors.mean()
    std_error = errors.std()
    summary = (f"Régression: RMSE = {rmse:.3f}, R2 = {r2:.3f}.\n"
               f"Erreur moyenne = {mean_error:.3f}, écart-type de l'erreur = {std_error:.3f}.")
    return summary

def summarize_shap_values(shap_values, feature_names, top_k=5):
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    indices = np.argsort(mean_abs_shap)[::-1][:top_k]
    summary = "Variables qui influencent le plus la prédiction (SHAP) :\n"
    for i in indices:
        summary += f"- {feature_names[i]} avec une importance moyenne absolue de {mean_abs_shap[i]:.4f}\n"
    return summary

def generate_force_plot_base64(shap_values, features):
    plt.figure(figsize=(10, 1))  # Ajuste la hauteur selon le besoin
    shap.plots.force(shap_values, matplotlib=True)
    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()
    return image_base64

def shap_plot_to_base64(plot_func, *args, **kwargs):
    plt.figure()
    plot_func(*args, **kwargs, show=False)
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()
    return image_base64



def generate_rmse_error_histogram_base64(y_true, y_pred):
    errors = y_pred - y_true  # or y_true - y_pred depending on your interpretation
    plt.figure(figsize=(8, 6))
    sns.histplot(errors, bins=30, kde=True, color='orange', stat='density')
    plt.axvline(0, color='blue', linestyle='--', label='Zero error')
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Error (Predicted - Actual)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()

    # Encode as base64 image
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    return image_base64

def generate_feature_importance_plot(importances):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances['importance'], y=importances.index, palette='viridis')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    return image_base64

def generate_metric_plot(metric_name, y_true=None, y_pred=None, y_proba=None, class_labels=None):
    fig, ax = plt.subplots()
    metric_name = metric_name.lower()

    if metric_name == 'accuracy':
        # Display confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=class_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title("Confusion Matrix")

    elif metric_name == 'roc_auc' and y_proba is not None:
        pos_label = class_labels[1] if class_labels else 1
        fpr, tpr, _ = roc_curve(y_true, y_proba, pos_label=pos_label)
        ax.plot(fpr, tpr, label='ROC Curve')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()

    elif metric_name == 'f1_macro':
        # F1 score bar plot per class
        f1_per_class = f1_score(y_true, y_pred, average=None, labels=class_labels)
        ax.bar(class_labels, f1_per_class, color='lightgreen')
        ax.set_title("F1 Score per Class")
        ax.set_ylabel("F1 Score")

    elif metric_name == 'rmse':
        ax.scatter(y_true, y_pred, alpha=0.5, color='coral', label='Predictions')
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        rmse_value = mean_squared_error(y_true, y_pred) ** 0.5
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x')
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"Predictions vs Actual (RMSE = {rmse_value:.4f})")
        ax.legend()

    else:
        ax.text(0.5, 0.5, f"No plot available for {metric_name}", ha='center', va='center')
        ax.set_axis_off()

    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return image_base64

def train_model(df, target_column, time_limit, save_path, include_shap):
    try:
        save_path = os.path.abspath(save_path)

        if df[target_column].isnull().any():
            raise UserError("error_missing_target_values")

        predictor = TabularPredictor(label=target_column, path=save_path).fit(df, time_limit=time_limit)

        model = predictor._trainer.load_model(predictor._trainer.model_best)

        shap_summary_plot = None

        feature_importance_df = predictor.feature_importance(df)
        feature_importance_plot = generate_feature_importance_plot(feature_importance_df)

        leaderboard = predictor.leaderboard(silent=True)
        best_model = predictor.model_best
        task_type = predictor.problem_type

        y_test = df[target_column]
        y_pred = predictor.predict(df)
        y_proba = predictor.predict_proba(df) if task_type == 'binary' else None
        class_labels = predictor.class_labels if hasattr(predictor, 'class_labels') else None

        feature_importance_summary = summarize_feature_importance(feature_importance_df)

        perf_data = {}
        summary = ""

        if task_type == 'binary':
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba[class_labels[1]])
            perf_data = {
                'accuracy': {
                    'value': acc,
                    'plot': generate_metric_plot('accuracy', y_true=y_test, y_pred=y_pred, class_labels=class_labels)
                },
                'roc_auc': {
                    'value': auc,
                    'plot': generate_metric_plot('roc_auc', y_true=y_test, y_proba=y_proba[class_labels[1]], class_labels=class_labels)
                }
            }
            summary = f"Tâche détectée : classification binaire\n\n"
            summary += f"Modèle sélectionné : {best_model}\n"
            summary += f"Temps d'entraînement : {float(leaderboard['fit_time'].sum()):.2f} seconds\n\n"
            summary += summarize_confusion_matrix(y_test, y_pred, class_labels)
            summary += summarize_roc_auc(y_test, y_proba[class_labels[1]], pos_label=class_labels[1])
            summary += f"Résultat de la métrique accuracy : {perf_data['accuracy']['value']:.4f}\n"
            summary += f"Résultat de la métrique ROC AUC : {perf_data['roc_auc']['value']:.4f}\n"

        elif task_type == 'multiclass':
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            perf_data = {
                'accuracy': {
                    'value': acc,
                    'plot': generate_metric_plot('accuracy', y_true=y_test, y_pred=y_pred, class_labels=class_labels)
                },
                'f1_macro': {
                    'value': f1,
                    'plot': generate_metric_plot('f1_macro', y_true=y_test, y_pred=y_pred, class_labels=class_labels)
                }
            }
            summary = f"Tâche détectée : classification multiclasse\n\n"
            summary += f"Modèle sélectionné : {best_model}\n"
            summary += f"Temps d'entraînement : {float(leaderboard['fit_time'].sum()):.2f} seconds\n\n"
            summary += summarize_confusion_matrix(y_test, y_pred, class_labels)
            summary += summarize_f1_per_class(y_test, y_pred, class_labels)
            summary += f"Résultat de la métrique accuracy : {perf_data['accuracy']['value']:.4f}\n"
            summary += f"Résultat de la métrique F1 macro : {perf_data['f1_macro']['value']:.4f}\n"

        elif task_type == 'regression':
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            perf_data = {
                'r2': {
                    'value': r2
                },
                'rmse': {
                    'value': rmse,
                    'plot': generate_metric_plot('rmse', y_true=y_test, y_pred=y_pred),
                    'plot_hist': generate_rmse_error_histogram_base64(y_true=y_test, y_pred=y_pred)
                }
            }
            summary = f"Tâche détectée : regression\n\n"
            summary += f"Modèle sélectionné : {best_model}\n"
            summary += f"Temps d'entraînement : {float(leaderboard['fit_time'].sum()):.2f} seconds\n\n"
            summary += summarize_regression_metrics(y_test, y_pred)
            summary += f"Résultat de la métrique R2 : {perf_data['r2']['value']:.4f}\n"
            summary += f"Résultat de la métrique RMSE : {perf_data['rmse']['value']:.4f}"

        score_metric = predictor.eval_metric.name if hasattr(predictor.eval_metric, 'name') else str(predictor.eval_metric)

        results = {
            'best_model': best_model,
            'train_time': float(leaderboard['fit_time'].sum()),
            'task_type': task_type,
            'score_metric': score_metric,
            'metrics': perf_data,
            'feature_importance_plot': feature_importance_plot,
            'model_path': save_path,
            'leaderboard': leaderboard.to_dict(orient='records'),
            'summary_LLM' : summary
        }

        return results

    except UserError:
        raise
    except Exception:
        raise UserError("error_unexpected_training")

    
def generate_shap_plot(model_path, df, target_column):
    try:
        predictor = TabularPredictor.load(model_path)
        model = predictor._trainer.load_model(predictor._trainer.model_best)

        if target_column not in df.columns:
            raise UserError("error_missing_target_column")

        X = df.drop(columns=[target_column])

        if predictor.problem_type in ["binary", "multiclass"]:
            explainer = shap.Explainer(model.predict_proba, X)
        else:
            explainer = shap.Explainer(model.predict, X)

        shap_values = explainer(X)

        shap_summary_plot = shap_plot_to_base64(shap.summary_plot, shap_values, X)

        return {
            'shap_summary_plot': shap_summary_plot
        }

    except UserError:
        raise
    except Exception:
        raise UserError("error_shap")



def plot_regression_distribution(y_pred):
    plt.figure(figsize=(8, 6))
    sns.histplot(y_pred, bins=30, kde=True, color='skyblue')
    plt.title("Distribution of Predicted Values")
    plt.xlabel("Predicted Values")
    plt.ylabel("Frequency")
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()
    return image_base64

def plot_prediction_outliers(y_pred):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=y_pred, color="tomato")
    plt.title("Outlier Detection in Predictions")
    plt.xlabel("Predicted Value")
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()
    return image_base64

def plot_predicted_class_distribution(y_pred):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y_pred, palette='pastel')
    plt.title("Predicted Class Distribution")
    plt.xlabel("Predicted Class")
    plt.ylabel("Number of Occurrences")
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()
    return image_base64

def plot_prediction_confidence(y_proba_df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=y_proba_df, orient='h', palette='Set2')
    plt.title("Predicted Probability Distribution by Class")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Class")
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()
    return image_base64


def predict_model(csv_path, model_path):
    try:
        predictor = TabularPredictor.load(model_path)

        ext = os.path.splitext(csv_path)[1].lower()
        if ext == '.csv':
            df = pd.read_csv(csv_path)
        elif ext in ['.xls', '.xlsx', '.xlsm']:
            df = pd.read_excel(csv_path)
        elif ext == '.arff':
            import arff
            with open(csv_path, 'r') as f:
                arff_data = arff.load(f)
            df = pd.DataFrame(arff_data['data'], columns=[a[0] for a in arff_data['attributes']])
        else:
            raise UserError("error_unsupported_file_format")

        if predictor.label in df.columns:
            df = df.drop(columns=[predictor.label])

        expected_cols = predictor.feature_metadata.get_features()
        missing_cols = set(expected_cols) - set(df.columns)
        extra_cols = set(df.columns) - set(expected_cols)

        if missing_cols:
            raise UserError("error_missing_columns")

        if extra_cols:
            raise UserError("error_extra_columns")

        predictions = predictor.predict(df)
        task_type = predictor.problem_type
        output_data = {
            'predictions': predictions.reset_index().to_dict(orient='records'),
            'plots': {}
        }

        if task_type == 'regression':
            output_data['plots'] = {
                'distribution': plot_regression_distribution(predictions),
            }

        elif task_type == 'binary':
            y_proba = predictor.predict_proba(df)
            output_data['plots'] = {
                'class_distribution': plot_predicted_class_distribution(predictions),
                'confidence': plot_prediction_confidence(y_proba)
            }

        elif task_type == 'multiclass':
            y_proba = predictor.predict_proba(df)
            output_data['plots'] = {
                'class_distribution': plot_predicted_class_distribution(predictions),
                'confidence': plot_prediction_confidence(y_proba)
            }

        return output_data

    except UserError:
        raise
    except Exception:
        raise UserError("error_unexpected_prediction")
