# 🤖 AutoML Model Trainer

An end-to-end AutoML pipeline built with **Optuna**, **scikit-learn**, and **Plotly**, designed to automatically find the best machine learning model and hyperparameters for any tabular dataset. Ideal for developers, analysts, and data scientists who want to quickly benchmark and train high-performing models with minimal effort.

---

## 🚀 Features

- ✅ Upload any structured `.csv` dataset
- ⚙️ Automatically selects and tunes:
  - `RandomForestClassifier`
  - `LogisticRegression`
  - `SVC`
- 📊 Uses **Optuna** for hyperparameter optimization
- 🔬 3-fold cross-validation accuracy scoring
- 📈 Generates interactive optimization visualizations (Plotly)
- 💾 Saves the best model as `best_model.pkl`
- 🧠 Ready for CLI, Flask API, or full app integration

---

## 🛠 Tech Stack

| Component         | Tools Used                   |
|------------------|------------------------------|
| Language          | Python 3.10+                 |
| Machine Learning  | scikit-learn, Optuna         |
| Visualization     | Plotly, Optuna dashboards    |
| Deployment        | VSCode, GitHub, SQLite       |
| Optional API      | Flask                        |

---

## 📦 Installation

```bash
git clone https://github.com/aayushpawar11/AutoML-Model-Trainer.git
cd AutoML-Model-Trainer

python -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows

pip install -r requirements.txt
