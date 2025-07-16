import pandas as pd, numpy as np
from .utils import id_hash

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts the *raw* transaction DataFrame, returns **wallet‑level** feature table.

    ❶ Minimal schema assumed in df
       - `tx_timestamp` (int | str) POSIX or iso8601
       - `wallet`       (str)       user address
       - `action`       (str)       {deposit, borrow, repay, redeemunderlying, liquidationcall}
       - `amount_usd`   (float)     value at tx time (already USD‑normalised)
       - `asset`        (str)       token symbol
       - `health_factor`(float)     as emitted by Aave (NaN for non‑borrow events)

    ❷ You may add or drop columns below if your raw export uses slightly different keys.
    """
    df = df.rename(columns={
        "userWallet": "wallet",
        "timestamp": "tx_timestamp",
        "actionData.amount": "amount_raw",
        "actionData.assetPriceUSD": "asset_price_usd",
        "actionData.assetSymbol": "asset"
    })

    # Convert timestamp to datetime
    df["ts"] = pd.to_datetime(df["tx_timestamp"], unit="s", errors="coerce")

    # Convert raw string amount and price to float
    df["amount_usd"] = pd.to_numeric(df["amount_raw"], errors="coerce") * pd.to_numeric(df["asset_price_usd"], errors="coerce")
    
    # Some transactions may not have a price or amount; drop those
    df = df.dropna(subset=["amount_usd", "ts", "wallet", "action"])

    # Assign dummy health factor for compatibility (not present in dataset)
    df["health_factor"] = np.nan

    df.sort_values(["wallet", "ts"], inplace=True)

    # --- basic pre‑processing ---
    df["ts"] = pd.to_datetime(df["tx_timestamp"], errors="coerce")
    df.sort_values(["wallet", "ts"], inplace=True)

    # # ratio of borrow/repay to deposits, liquidation flags, 30‑day behaviour, etc.
    agg = {
        "amount_usd":               ["sum", "mean", "max"],
        "action":                   ["nunique", "count"],
        "health_factor":            ["min"]
    }

    f = df.groupby("wallet").agg(agg)
    f.columns = ["_".join(c) for c in f.columns]

    # === behavioural ratios ===
    act_counts = df.pivot_table(index="wallet",
                                columns="action",
                                values="amount_usd",
                                aggfunc="count",
                                fill_value=0)
    for a in ["deposit", "borrow", "repay", "redeemunderlying", "liquidationcall"]:
        if a not in act_counts.columns:
            act_counts[a] = 0

    f["pct_borrow_tx"]  = act_counts["borrow"]  / f["action_count"]
    f["pct_liquidated"] = act_counts["liquidationcall"] / f["action_count"]
    f["num_days_active"] = df.groupby("wallet")["ts"].apply(lambda s: s.dt.date.nunique())

    # time since first/last action
    first_last = df.groupby("wallet")["ts"].agg(["min", "max"])
    snapshot_date = df["ts"].max()
    f["age_days"]   = (snapshot_date - first_last["min"]).dt.days
    f["recent_days"]= (snapshot_date - first_last["max"]).dt.days

    # anonymise for plots
    f["wallet_id"] = f.index.map(id_hash)
    return f.reset_index(drop=False).set_index("wallet")
