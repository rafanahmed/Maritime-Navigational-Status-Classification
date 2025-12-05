from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier

def build_logistic_regression_model():
    return LogisticRegression(
        multi_class="multinomial", 
        solver="lbfgs", 
        max_iter=1000, 
        n_jobs=-1,
        class_weight="balanced"
    )

def build_random_forest_model(n_estimators: int = 300, max_depth: int | None = None, 
random_state: int = 42, class_weight: str = "balanced"):
    return RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=random_state,
        n_jobs=-1,
        class_weight=class_weight
    )

def build_xgboost(n_estimators: int = 300, learning_rate: float = 0.05, max_depth: int = 6,
subsample: float = 0.8, colsample_bytree: float = 0.8, random_state: int = 42, scale_pos_weight: int | None = None):
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist"
    )

print("Compilation complete")