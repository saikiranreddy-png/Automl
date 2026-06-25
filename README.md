Project Overview

AutoML360 is an intelligent automated machine learning pipeline designed to streamline the process of training, evaluating, and selecting ML models. It also integrates explainability features so users can understand why models make specific predictions. The project is deployed as a Streamlit web app for interactive usage.

Features

End-to-End Automation: Handles data preprocessing, train/test split, model training, evaluation, and comparison automatically.

Model Selection: Compares multiple algorithms (e.g., Logistic Regression,Gradient boost, Random Forest,) and recommends the best-performing model along with clustering tasks.

Explainability: Generates feature importance and interpretability visualizations to improve transparency.

Interactive Web App: Streamlit-based UI for real-time experimentation without coding.

Metrics Reporting: Computes performance metrics such as accuracy, precision, recall, and F1-score.

Tech Stack

Language: Python 3.x

Libraries: scikit-learn, pandas, numpy, matplotlib, seaborn, streamlit

Visualization & Explainability: SHAP / Matplotlib / Seaborn


Usage

Upload your dataset (CSV format).

Select target column and optionally choose models to train.

Run the pipeline to automatically:

Preprocess data

Train multiple models

Evaluate and compare performance

Generate explainability visualizations

Download model reports or interact with visualizations directly in the app.

How It Works

Preprocessing: Handles missing values, encoding categorical variables, and scaling numerical features.

Model Training: Loops through predefined ML models and trains them on the dataset.

Evaluation & Selection: Computes metrics (accuracy, precision, recall, F1-score) for all models and recommends the best one.

Explainability: Generates feature importance plots to show the influence of features on predictions.

Streamlit Deployment: Users interact via a clean UI, run experiments in seconds, and visualize results instantly.
