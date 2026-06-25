import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, mean_squared_error

from supervised import run_supervised
from clustering import run_clustering
from association import run_association
from explain import explain_instance

st.set_page_config(page_title="AutoML360", layout="wide")
st.title("🔮 AutoML360 : Automated Machine Learning with Explainability")

# ---------------------------
# UPLOAD DATASET
# ---------------------------
uploaded = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])
task = st.selectbox("⚡ Select Task", ["auto", "classification", "regression", "clustering", "association"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("### 👀 Data Preview")
    st.dataframe(df.head())

    target = None
    if task in ["auto", "classification", "regression"]:
        target = st.selectbox("🎯 Select Target Column", df.columns)

    if "automl_results" not in st.session_state:
        st.session_state["automl_results"] = None

    if st.button("🚀 Run AutoML"):
        st.info("Running AutoML — please wait...")
        try:
            if task in ["auto", "classification", "regression"]:
                if target is None:
                    st.error("❌ Please select a target column for supervised tasks.")
                else:
                    res = run_supervised(df, target=target, task_hint=task)
                    st.session_state["automl_results"] = ("supervised", res, df, target)
            elif task == "clustering":
                res = run_clustering(df)
                st.session_state["automl_results"] = ("clustering", res, df, None)
            elif task == "association":
                res = run_association(df)
                st.session_state["automl_results"] = ("association", res, df, None)

            st.success("✅ AutoML finished successfully!")
        except Exception as e:
            st.error(f"⚠️ AutoML failed: {e}")

    # ---------------------------
    # RESULTS DISPLAY
    # ---------------------------
    if st.session_state["automl_results"] is not None:
        kind, res, df_saved, target_saved = st.session_state["automl_results"]

        st.write("---")
        st.header("📊 AutoML Results")

        # ---------------------------
        # SUPERVISED RESULTS
        # ---------------------------
        if kind == "supervised":
            st.subheader("🔎 Supervised Learning (Classification/Regression)")

            # Compute final pipeline metric
            pipeline = res["best_pipeline"]
            model = res["model"]
            X_full = df_saved.drop(columns=[target_saved])
            y_full = df_saved[target_saved]

            if res["task"] == "classification":
                y_pred = pipeline.predict(X_full)
                final_score = f1_score(y_full, y_pred, average="macro")
                metric_name = "F1_macro"
            else:  # regression
                y_pred = pipeline.predict(X_full)
                final_score = mean_squared_error(y_full, y_pred, squared=False)  # RMSE
                metric_name = "RMSE"

            # Show Algorithm Comparison table
            st.write("### Algorithm Comparison")
            all_algos = []
            for algo, details in res["cv_results"].items():
                # Use CV score
                score = details.get(metric_name, details.get("mean_cv", np.nan))
                if res["task"] == "regression" and score < 0:
                    score = float(np.sqrt(-score))
                all_algos.append([algo, round(score, 3), metric_name])

            # Replace best algo score with final refit score
            for row in all_algos:
                if row[0] == res["best_algo"]:
                    row[1] = round(final_score, 3)

            df_algos = pd.DataFrame(all_algos, columns=["Algorithm", "Score", "Metric"])
            # Highlight best algorithm
            def highlight_best(row):
                return ['background-color: lightgreen' if row.Algorithm == res["best_algo"] else '' for _ in row]
            st.dataframe(df_algos.style.apply(highlight_best, axis=1))

            st.success(f"🏆 Best Algorithm: **{res['best_algo']}** with {metric_name} = {round(final_score,3)}")

            # Feature importance
            st.write("### 🔥 Feature Importance (Top 10)")
            if not res["importance"].empty:
                st.dataframe(res["importance"].head(10))
                fig, ax = plt.subplots(figsize=(8, 5))
                topn = res["importance"].head(10)
                ax.barh(topn["feature"], topn["importance"])
                ax.invert_yaxis()
                st.pyplot(fig)
            else:
                st.write("No feature importance available.")

            # Row explanation
            st.write("### 🧾 Explain a Row Prediction")
            row_idx = st.number_input("Row index", min_value=0, max_value=len(df_saved)-1, value=0)
            if st.button("Explain selected row"):
                inst = df_saved.drop(columns=[target_saved]).iloc[[row_idx]]
                explanation = explain_instance(pipeline, model, df_saved.drop(columns=[target_saved]), inst, top_k=3)
                st.markdown(explanation)

        # ---------------------------
        # CLUSTERING RESULTS
        # ---------------------------
        elif kind == "clustering":
            st.subheader("🔎 Clustering (Unsupervised)")

            st.write("### Algorithm Comparison")
            all_clusters = []
            for algo, details in res["algo_results"].items():
                if "silhouette" in details:
                    all_clusters.append([algo, round(details["silhouette"], 3)])
                else:
                    all_clusters.append([algo, "Error"])
            st.dataframe(pd.DataFrame(all_clusters, columns=["Algorithm", "Silhouette Score"]))

            st.success(f"🏆 Best Algorithm: **{res['best_algo']}** with Silhouette Score = {round(res['metric'], 3)}")

            st.write("### 📌 Cluster Assignments (first 20 rows)")
            df_saved["Cluster"] = res["clusters"]
            st.dataframe(df_saved.head(20))

        # ---------------------------
        # ASSOCIATION RESULTS
        # ---------------------------
        elif kind == "association":
            st.subheader("🔎 Association Rule Mining")
            st.write("### Frequent Itemsets")
            st.dataframe(res["frequent_itemsets"].head(10))
            st.write("### Association Rules")
            st.dataframe(res["rules"].head(10))
