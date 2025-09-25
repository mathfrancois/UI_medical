from groq import Groq
from utils.prompts import get_system_prompt_chat, get_keywords_by_lang, get_system_prompt_preprocessing, get_system_prompt_explanation_plot
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle, SimpleDocTemplate, Spacer, Image, PageBreak
from reportlab.lib.enums import TA_LEFT
from datetime import datetime
import io
import base64
import re

# Retrieve the GROQ API key from environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
print(repr(GROQ_API_KEY))

def clean_markdown(text):
    # Remove markdown bold and italic markers from text
    return re.sub(r'(\*\*|\*)', '', text)

def handle_chat_request(data):
    """
    Handles a chat request by preparing the context and sending it to the Groq API.
    It builds the conversation history, attaches dataset previews, stats, plots, etc.
    """
    user_input = data.get('message', '')
    if not user_input:
        raise ValueError("error_empty_message")

    lang = data.get('lang', 'en')
    summary = data.get('summary')
    dataset = data.get('markdown_preview')
    stats = data.get('stats')
    shap_plot = data.get('shap_summary_plot')
    plot_predictions = data.get('plots_prediction_results')
    training_pred_plot = data.get('png_result_training_for_prediction')
    # Start with a system prompt for the chat
    messages = [{"role": "system", "content": get_system_prompt_chat(lang)}]

    # If the user asks about the dataset but none is provided, add a warning message
    if dataset is None and any(kw in user_input.lower() for kw in get_keywords_by_lang(lang)):
        messages.append({
            "role": "assistant",
            "content": "Je ne vois pas encore de dataset ou de résultats d'entraînement. Pour que je puisse vous aider, vous devez d’abord lancer un entraînement via l’interface avec vos données."
        })

    # Append dataset preview, stats, training summary if available
    append_if_exists(messages, "Aperçu du dataset fourni :", dataset)
    append_if_exists(messages, "Statistiques du dataset :", stats)
    if summary:
        append_if_exists(messages, "Résumé des résultats d'entraînement :", summary.get('text'))

    # Add metric plots from summary if present
    if summary:
        add_metric_plots_from_summary(messages, summary)

    # Add SHAP plot if present
    if shap_plot:
        add_image_message(messages, "Voici le graphe SHAP de l'entraînement.", shap_plot)

    # Add prediction result plots if present
    if plot_predictions:
        for name, plot in plot_predictions.items():
            add_image_message(messages, f"Voici un des graphes {name} des résultats de prédiction.", plot)

    # Add training prediction plots if present
    if training_pred_plot:
        for name, plot in training_pred_plot.items():
            add_image_message(messages, f"Voici un des graphes {name} qui montre les résultats de l'entrainement du modèle qui sert à faire les prédictions.", plot)

    # Add the user's message at the end
    messages.append({"role": "user", "content": user_input})

    # Send the conversation to the Groq API and get the response
    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages,
        temperature=0.7,
        max_completion_tokens=1024,
        top_p=1,
        stream=False
    )

    return completion.choices[0].message.content

def append_if_exists(messages, title, content):
    # Append a message to the conversation if content is not empty
    if content:
        messages.append({"role": "user", "content": f"{title}\n{content}"})

def add_image_message(messages, caption, base64_str):
    # Add an image (base64-encoded) with a caption to the conversation
    messages.append({"role": "user", "content": [
        {"type": "text", "text": caption},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_str}"}}
    ]})

def add_metric_plots_from_summary(messages, summary):
    # Add feature importance plot and metric plots from the summary to the conversation
    if "feature_importance_plot" in summary:
        add_image_message(messages, "Voici le graphe de l'importance des variables.", summary["feature_importance_plot"])
    for metric_name, metric_obj in summary.get("metrics_plot", {}).items():
        for plot_type in ['plot', 'plot_hist']:
            if plot_type in metric_obj:
                base64_plot = metric_obj[plot_type]
                if base64_plot:
                    add_image_message(messages, f"Voici le graphe {metric_name} ({plot_type}).", base64_plot)

def build_preprocessing_summary(preprocessing_summary: dict, lang='en') -> str:
    """
    Generates a natural language summary of the preprocessing steps using the Groq API.
    """
    client = Groq(api_key=GROQ_API_KEY)

    system_prompt = get_system_prompt_preprocessing()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(preprocessing_summary)}  
    ]

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages,
        temperature=0.5,
        max_completion_tokens=512,
        top_p=1,
        stream=False
    )

    return response.choices[0].message.content

def generate_explanation_plot(plot = None, name = None, value = None):
    """
    Generates an explanation for a given plot and metric value using the Groq API.
    """
    messages = [{"role": "system", "content": get_system_prompt_explanation_plot()}]
    if plot and value:
        add_image_message(messages, f"Here the graph {name} to explain and the value of the metric {value}", plot)
    elif plot:
        add_image_message(messages, f"Here the graph {name} to explain.", plot)
    else:
        messages.append({"role": "user", "content": f"Please explain the metric {name} with the value {value}."})
    
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages,
        temperature=0.5,
        max_completion_tokens=512,
        top_p=1,
        stream=False
    )
    raw_text = response.choices[0].message.content
    return clean_markdown(raw_text)

def generate_pdf(summary_results, dataset_preview, dataset_stats, preprocessing_summary,
                 target_column, dataset_name, shap_plot, output_dir="pdf_summary"):
    """
    Generates a PDF summary report of the training results, including dataset preview,
    statistics, preprocessing, metrics, plots, and explanations.
    """
    # === Prepare output directory ===
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"{dataset_name}_{timestamp}.pdf"
    output_path = os.path.join(output_dir, pdf_filename)

    # === Create PDF document ===
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=60, bottomMargin=40)
    styles = getSampleStyleSheet()
    elements = []

    # Title style and content
    title_style = styles['Heading1']
    title_style.alignment = 1

    today_str = datetime.now().strftime("%B %d, %Y")
    title_text = f"Training Summary - {dataset_name} ({today_str})"
    elements.append(Paragraph(title_text, title_style))
    elements.append(Spacer(1, 20))

    # === 1. Dataset Overview ===
    elements.append(Paragraph("1. Dataset Overview", styles['Heading2']))
    elements.append(Spacer(1, 12))

    # Dataset preview (markdown-like)
    elements.append(Paragraph("<b>Dataset Preview (first 5 rows)</b>", styles['Normal']))
    
    # Parse the markdown preview into a table
    preview_lines = dataset_preview.strip().split('\n')
    preview_data = [
        line.strip('|').split('|')
        for line in preview_lines
        if '|' in line and not re.match(r'^\s*-{2,}', line.strip().replace('|', '').strip())
    ]
    preview_data = [[cell.strip() for cell in row] for row in preview_data]

    # Detect the number of columns
    max_columns = len(preview_data[0])

    # Dynamically adjust column widths
    page_width = A4[0] - doc.leftMargin - doc.rightMargin
    col_width = page_width / max_columns
    col_widths = [col_width] * max_columns

    # Apply word wrap to table cells using Paragraph
    cell_style = ParagraphStyle(
        name='TableCell',
        fontName='Helvetica',
        fontSize=7 if max_columns > 8 else 9,
        leading=10,
        alignment=TA_LEFT,
        wordWrap='CJK',  # force word wrap
    )

    # Replace cell values with Paragraphs for wrapping
    wrapped_data = []
    for row in preview_data:
        wrapped_row = [Paragraph(cell, cell_style) for cell in row]
        wrapped_data.append(wrapped_row)

    # Create the table for dataset preview
    table = Table(wrapped_data, colWidths=col_widths, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 12))

    # Dataset statistics table
    elements.append(Paragraph("<b>Dataset Statistics</b>", styles['Normal']))
    stat_table_data = [["Column", "Type", "Missing", "Mean", "Std"]]
    for col in dataset_stats:
        stat_table_data.append([
            col['name'],
            col['type'],
            str(col['missing']),
            col['mean'],
            col['std']
        ])
    stat_table = Table(stat_table_data, hAlign='LEFT')
    stat_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(stat_table)
    elements.append(Spacer(1, 12))

    # Preprocessing summary section
    elements.append(Paragraph("<b>Preprocessing Summary</b>", styles['Normal']))
    elements.append(Paragraph(preprocessing_summary, styles['Normal']))
    elements.append(Spacer(1, 12))

    # Target column section
    elements.append(Paragraph(f"<b>Target Column:</b> {target_column}", styles['Normal']))
    elements.append(Spacer(1, 20))

    elements.append(PageBreak())

    # === 2. Training Results ===
    elements.append(Paragraph("2. Training Results", styles['Heading2']))
    elements.append(Spacer(1, 12))

    # Add task type, best model, and training time
    elements.append(Paragraph(f"- Detected Task Type: <b>{summary_results['task_type']}</b>", styles['Normal']))
    elements.append(Paragraph(f"- Best Model: <b>{summary_results['best_model']}</b>", styles['Normal']))
    elements.append(Paragraph(f"- Training Time: <b>{round(summary_results['train_time'], 2)} seconds</b>", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Metrics table
    metrics_data = [["Metric", "Score"]]
    for metric, data in summary_results['metrics_plot'].items():
        metrics_data.append([metric.upper(), f"{data['value']:.4f}"])
    metric_table = Table(metrics_data, hAlign='LEFT')
    metric_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(Paragraph("<b>Evaluation Metrics</b>", styles['Normal']))
    elements.append(metric_table)
    elements.append(Spacer(1, 20))

    # === 3. Visual Results ===
    elements.append(Paragraph("3. Training Visualizations", styles['Heading2']))
    elements.append(Spacer(1, 12))

    # === Visualization of metrics (plots if available) ===
    for metric, data in summary_results['metrics_plot'].items():
        # Title and score
        score = round(data.get('value', 0.0), 4)
        elements.append(Paragraph(f"<b>{metric.upper()} Score</b>: {score}", styles['Normal']))
        
        # Display all available plots (plot and/or plot_hist)
        for plot_type in ['plot', 'plot_hist']:
            if plot_type in data:
                image_data = base64.b64decode(data[plot_type])
                image_stream = io.BytesIO(image_data)
                img = Image(image_stream, width=400, height=250)
                elements.append(img)
                elements.append(Spacer(1, 6))

            # Generate and add explanation for each plot
            explanation = generate_explanation_plot(plot=data.get(plot_type), name=metric, value=score)
            if explanation:
                elements.append(Paragraph(explanation, styles['Normal']))

        elements.append(Spacer(1, 18))

    # Feature importance plot and explanation
    if 'feature_importance_plot' in summary_results:
        elements.append(Paragraph("<b>Feature Importance</b>", styles['Normal']))
        feature_img_data = base64.b64decode(summary_results['feature_importance_plot'])
        feature_img_stream = io.BytesIO(feature_img_data)
        feature_img = Image(feature_img_stream, width=400, height=250)
        elements.append(feature_img)

        explanation = generate_explanation_plot(plot=summary_results['feature_importance_plot'], name="Feature Importance", value=None)
        if explanation:
            elements.append(Paragraph(explanation, styles['Normal']))

        elements.append(Spacer(1, 18))
    if shap_plot:
        elements.append(Paragraph("<b>SHAP Summary Plot</b>", styles['Normal']))
        shap_img_data = base64.b64decode(shap_plot)
        shap_img_stream = io.BytesIO(shap_img_data)
        shap_img = Image(shap_img_stream, width=400, height=250)
        elements.append(shap_img)

        explanation = generate_explanation_plot(plot=shap_plot, name="SHAP Summary", value=None)
        if explanation:
            elements.append(Paragraph(explanation, styles['Normal']))

    # Build the PDF document and return the buffer
    doc.build(elements)
    buffer.seek(0)  
    return buffer
