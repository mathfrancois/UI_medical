# Medical AutoML Assistant

This project provides a user-friendly web interface for medical professionals to easily train, evaluate, and use machine learning models without requiring any background in data science or coding. The system is powered by [AutoGluon](https://auto.gluon.ai/) and includes a conversational assistant (LLM) to guide the user through each step.

---

## Project Structure

* **app.py**: Main Flask application.
* **model\_utils.py**: Logic for training, predicting, and plotting.
* **cleanup.py**: Utility script to clean temporary files.
* **templates/**: HTML interface (interface.html).
* **static/**: CSS and JS files (frontend logic and style).
* **logs/**: Stores application logs.
* **requirements.txt**: Lists all Python dependencies.

---

## Installation and Execution

### Prerequisites

* Python 3.8+
* pip
* Recommended: Virtual environment (venv)

---

### Installation Steps

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd UI_medical
```

2. **Create and activate a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

### Run the Application

```bash
python app.py
```

Then open your browser at: `http://127.0.0.1:5000/`

---

## How to Use

### Training Tab

1. **Upload a CSV training dataset**
2. **Preview the dataset** and optionally remove columns and fill missing value
3. **Select the target column** (to predict)
4. **Set a training time limit (seconds)**
5. **Click to launch training**
6. **Review the results**:

   * Table of performance metrics
   * Model leaderboard
   * Confusion matrix and ROC curve (for classification)
   * Feature importance from AutoGluon and SHAP beeswarm plots
7. **Download the trained model (.zip)**

### Prediction Tab

1. **Upload a trained model (.zip)**
2. **Upload a new dataset (CSV)**
3. **Run predictions**
4. **Preview results and download the output CSV**

---

## Built-in Chatbot (LLM Assistant)

The assistant helps medical staff by:

* Explaining how to use the interface (column selection, preprocessing, etc.)
* Describing performance metrics and graphs
* Suggesting improvements to the dataset
* Helping understand why a model performs well or poorly
* Interpreting SHAP plots and feature importance

This assistant is designed to work with **non-technical users**.

---

## Example Workflow

1. Upload `patients.csv`
2. Choose `disease_outcome` as the target
3. Set training time to 300 seconds
4. Launch training
5. Review SHAP/ROC/matrix results
6. Download the best model
7. Upload `new_patients.csv` for prediction
8. Download predictions

---

## License

---

## Contact

