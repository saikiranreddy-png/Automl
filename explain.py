# explain.py
import numpy as np
import pandas as pd
import warnings
try:
    import shap
    SHAP = True
except Exception:
    SHAP = False
def explain_instance(pipeline, model, X_train: pd.DataFrame, instance: pd.DataFrame, top_k: int = 3) -> str:
    """
    Return a conversational, human-readable explanation for a single-row instance.
    """
    if instance.shape[0] != 1:
        raise ValueError("Instance must be single-row DataFrame")

    pre = pipeline.named_steps.get("preprocessor") or pipeline.named_steps.get("preprocess")
    try:
        feat_names = get_feature_names(pre, X_train.columns.tolist()) if pre is not None else X_train.columns.tolist()
    except Exception:
        feat_names = X_train.columns.tolist()

    # Try SHAP (if available)
    if SHAP:
        try:
            explainer = None
            if hasattr(model, "feature_importances_"):
                explainer = shap.TreeExplainer(model)
                transformed = pre.transform(instance) if pre is not None else instance.values
                shap_values = explainer.shap_values(transformed)

                if isinstance(shap_values, list):  # multiclass
                    pred = pipeline.predict(instance)[0]
                    class_idx = int(pred) if str(pred).isdigit() else 0
                    shap_arr = shap_values[class_idx][0]
                else:
                    shap_arr = shap_values[0]

                idx_top = np.argsort(np.abs(shap_arr))[::-1][:top_k]
                pred_label = pipeline.predict(instance)[0]

                msgs = [f"The model believes the outcome is **{pred_label}**."]
                msgs.append("Here’s why:")

                for i in idx_top:
                    fname = feat_names[i] if i < len(feat_names) else f"feature_{i}"
                    contrib = shap_arr[i]
                    if contrib > 0:
                        msgs.append(f"- The value of **{fname}** strongly pushed the decision towards this result.")
                    else:
                        msgs.append(f"- The value of **{fname}** slightly held back the prediction.")

                msgs.append("Overall, these features worked together to influence the final choice.")
                return "\n".join(msgs)
        except Exception as e:
            warnings.warn("SHAP conversational explanation failed: " + str(e))

    # Fallback: feature_importances_ or coef_
    try:
        if hasattr(model, "feature_importances_"):
            imps = np.array(model.feature_importances_)
            top_idx = np.argsort(imps)[::-1][:top_k]
            pred = pipeline.predict(instance)[0]
            msgs = [f"The model predicts **{pred}**."]
            msgs.append("The most influential reasons were:")
            for i in top_idx:
                fname = feat_names[i] if i < len(feat_names) else f"feature_{i}"
                msgs.append(f"- **{fname}**, which played a big role in shaping this outcome.")
            return "\n".join(msgs)

        elif hasattr(model, "coef_"):
            coef = np.array(model.coef_)
            if coef.ndim > 1:
                coef_vals = np.mean(np.abs(coef), axis=0)
            else:
                coef_vals = np.abs(coef)
            top_idx = np.argsort(coef_vals)[::-1][:top_k]
            pred = pipeline.predict(instance)[0]
            msgs = [f"The model thinks the answer is **{pred}**."]
            msgs.append("It mainly relied on:")
            for i in top_idx:
                fname = feat_names[i] if i < len(feat_names) else f"feature_{i}"
                msgs.append(f"- **{fname}**, which strongly guided the decision.")
            return "\n".join(msgs)
    except Exception as e:
        warnings.warn("Fallback conversational explanation failed: " + str(e))

    return "The model gave a result, but I couldn’t generate a natural explanation for it."
