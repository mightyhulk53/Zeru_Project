#!/usr/bin/env python3
"""
One‑command runner:

    python score_wallets.py --input raw_user_transactions.json --output wallet_scores.csv
"""

import argparse, pathlib, pandas as pd
from src.utils import set_seeds, read_jsonl, init_logger
from src.features import build_features
from src.model import train_model, predict_scores

def main():
    init_logger(); set_seeds()

    parser = argparse.ArgumentParser(description="Aave wallet credit scoring")
    parser.add_argument("--input",  required=True, help="Raw JSON/JSONL path")
    parser.add_argument("--output", default="wallet_scores.csv", help="Where to save scores")
    parser.add_argument("--retrain", action="store_true", help="Force re‑train a fresh model")
    args = parser.parse_args()

    print("⏳ Loading raw tx data…")
    df_raw = read_jsonl(args.input)
    print(f"Loaded {len(df_raw):,} transactions.")

    print("⚙️  Building features…")
    df_feat = build_features(df_raw)

    model_path = pathlib.Path("credit_model.pkl")
    if args.retrain or not model_path.exists():
        print("🧠 Training new model…")
        train_model(df_feat, save_path=model_path)

    print("🔮 Scoring wallets…")
    df_scores = predict_scores(df_feat, model_path=model_path)
    df_scores.to_csv(args.output, index=False)
    print(f"✅ Done. Scores written to {args.output}")

if __name__ == "__main__":
    main()
