# association.py
import pandas as pd
from typing import Dict, Any
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND = True
except Exception:
    MLXTEND = False

def run_association(df: pd.DataFrame, min_support: float = 0.2, lift_threshold: float = 1.0) -> Dict[str, Any]:
    if not MLXTEND:
        raise ImportError("mlxtend.frequent_patterns required. pip install mlxtend")

    X = df.copy().fillna(0)
    dummies = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            # discretize into 4 bins
            try:
                binned = pd.qcut(X[col].rank(method="first"), q=4, labels=False, duplicates="drop")
                db = pd.get_dummies(binned, prefix=f"{col}_bin")
                dummies.append(db)
            except Exception:
                db = (X[col] != 0).astype(int).to_frame(name=f"{col}_present")
                dummies.append(db)
        else:
            db = pd.get_dummies(X[col].astype(str), prefix=col)
            dummies.append(db)

    if len(dummies) == 0:
        raise RuntimeError("No columns processed for association rules.")

    bin_df = pd.concat(dummies, axis=1).astype(int)
    bin_df = (bin_df > 0).astype(int)

    freq_items = apriori(bin_df, min_support=min_support, use_colnames=True)
    rules = association_rules(freq_items, metric="lift", min_threshold=lift_threshold)

    return {"task": "association", "frequent_itemsets": freq_items, "rules": rules}
