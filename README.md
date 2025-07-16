# 🔍 Aave V2 Wallet Credit Scoring

This project generates **credit scores (0–1000)** for wallets based on historical transaction-level data from the **Aave V2** protocol. The scores reflect responsible or risky usage behavior, helping distinguish bots, exploiters, and good-faith users.

---

## 📦 Contents

- [🔧 Features](#-features)
- [🧠 Labeling Logic](#-labeling-logic)
- [⚙️ Architecture](#️-architecture)

---

## 🔧 Features

From raw transaction logs, we engineer a wallet-level feature table:

| Feature Name             | Description                                           |
|--------------------------|-------------------------------------------------------|
| `amount_usd_sum`         | Total USD volume transacted by wallet                |
| `action_nunique`         | Unique types of actions used (e.g. deposit, borrow)  |
| `pct_borrow_tx`          | % of transactions that are borrows                   |
| `pct_liquidated`         | % of transactions that are liquidation calls         |
| `num_days_active`        | Number of unique days the wallet was active          |
| `age_days`               | Time since first recorded transaction                |
| `recent_days`            | Time since last activity                             |
| `health_factor_min`      | Lowest recorded Aave health factor (if any)          |

These features are extracted in [`src/features.py`](src/features.py).

---

## 🧠 Labeling Logic

We use **weak supervision** to label some wallets as `good` or `risky` based on heuristics:

- ✅ **Good (label = 1)**:
  - Never liquidated (`pct_liquidated = 0`)
  - Minimum health factor > 1.2

- ❌ **Risky (label = 0)**:
  - Ever liquidated (`pct_liquidated > 0`) or
  - Health factor < 1.0

Wallets that fall between these extremes are **unlabeled** and ignored during model training.

---

## ⚙️ Architecture

```text
raw_user_transactions.json
        ↓
     [utils.py]
   JSON → pandas DataFrame
        ↓
   [features.py]
 wallet-level feature table
        ↓
    [model.py]
 weak-labels + LightGBM classifier
        ↓
 predicted probabilities × 1000
        ↓
wallet_scores.csv
