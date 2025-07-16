import joblib, pathlib, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

TARGET_COL = "target_is_good"

def label_function(df_feat: pd.DataFrame) -> pd.Series:
    """
    Weak‑supervision heuristic:
      good  = never liquidated *and* pct_liquidated == 0 and health_factor_min > 1.2
      risky = ever liquidated  OR health_factor_min < 1.0
    """
    good = (df_feat["pct_liquidated"] == 0) & (df_feat["health_factor_min"].fillna(10) > 1.2)
    risky = (df_feat["pct_liquidated"] > 0) | (df_feat["health_factor_min"].fillna(10) < 1.0)
    y = np.where(good, 1,
            np.where(risky, 0, np.nan))
    return pd.Series(y, index=df_feat.index)

def train_model(df_feat: pd.DataFrame, seed: int = 42, save_path: str | pathlib.Path = "credit_model.pkl"):
    y = label_function(df_feat)
    usable = ~y.isna()
    X_train, X_val, y_train, y_val = train_test_split(df_feat.loc[usable].drop(columns=[TARGET_COL], errors="ignore"),
                                                      y[usable],
                                                      test_size=0.2,
                                                      random_state=seed,
                                                      stratify=y[usable])
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val   = lgb.Dataset(X_val,   label=y_val)

    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=31,
        seed=seed,
        feature_pre_filter=False,
    )

    booster = lgb.train(params,
                        lgb_train,
                        valid_sets=[lgb_train, lgb_val],
                        num_boost_round=500,
                        early_stopping_rounds=25,
                        verbose_eval=100)

    auc = roc_auc_score(y_val, booster.predict(X_val))
    print(f"Validation AUC: {auc:0.4f}")
    joblib.dump(booster, save_path)
    return booster

def predict_scores(df_feat: pd.DataFrame, model_path: str | pathlib.Path = "credit_model.pkl") -> pd.DataFrame:
    booster = lgb.Booster(model_file=str(model_path))
    proba = booster.predict(df_feat)
    # Map 0‑1 probability into 0‑1000 credit score (higher=better).
    scores = np.round(proba * 1000).astype(int)
    return pd.DataFrame({"wallet": df_feat.index, "credit_score": scores})
