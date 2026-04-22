"""
batch_predict.py — Bulk Prediction Script for Patient No-Show System.

Usage:
    python batch_predict.py --input <path_to_input.csv> --output <path_to_output.csv>

The input CSV must have these columns:
    PatientId, AppointmentID, Gender, ScheduledDay, AppointmentDay, Age,
    Neighbourhood, Scholarship, Hipertension, Diabetes, Alcoholism, Handcap, SMS_received

Example:
    python batch_predict.py --input Dataset/noshowappointments-kagglev2-may-2016.csv --output results/predictions.csv
"""

import argparse
import os
import sys
import time
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import joblib

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("batch_predict.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join("models", "best_model_pipeline.joblib")

# ── Feature Engineering (mirrors model_pipeline.py) ───────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering as done during training."""
    df = df.copy()

    # Remove invalid ages
    initial_len = len(df)
    df = df[df["Age"] >= 0]
    removed = initial_len - len(df)
    if removed > 0:
        logger.warning(f"Removed {removed} rows with invalid Age.")

    # Parse dates
    df["ScheduledDay"]   = pd.to_datetime(df["ScheduledDay"],   errors="coerce")
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"], errors="coerce")

    # Drop rows where dates couldn't be parsed
    null_dates = df[["ScheduledDay", "AppointmentDay"]].isnull().any(axis=1).sum()
    if null_dates > 0:
        logger.warning(f"Dropping {null_dates} rows with unparseable dates.")
        df = df.dropna(subset=["ScheduledDay", "AppointmentDay"])

    # Engineered features
    df["WaitingTime"] = (
        df["AppointmentDay"].dt.normalize() - df["ScheduledDay"].dt.normalize()
    ).dt.days.apply(lambda x: max(x, 0))

    df["AppointmentDayOfWeek"] = df["AppointmentDay"].dt.dayofweek
    df["AppointmentMonth"]     = df["AppointmentDay"].dt.month
    df["IsWeekend"]            = df["AppointmentDayOfWeek"].apply(lambda x: 1 if x >= 5 else 0)

    return df


def build_input_df(df: pd.DataFrame, numeric_features: list, cat_features: list) -> pd.DataFrame:
    """Select and fill the expected input columns."""
    all_expected = numeric_features + cat_features

    # Fill missing optional columns with defaults
    defaults = {
        "Age": 30, "Scholarship": 0, "Hipertension": 0, "Diabetes": 0,
        "Alcoholism": 0, "Handcap": 0, "SMS_received": 0,
        "WaitingTime": 5, "AppointmentDayOfWeek": 1,
        "AppointmentMonth": 5, "IsWeekend": 0,
        "Gender": "F", "Neighbourhood": "CENTRO"
    }
    for col in all_expected:
        if col not in df.columns and col in defaults:
            df[col] = defaults[col]
            logger.warning(f"Column '{col}' missing — filled with default: {defaults[col]}")

    return df[all_expected]


def assign_risk(prob: float) -> str:
    """Assign risk label based on probability threshold."""
    if prob < 0.35:
        return "Low"
    elif prob < 0.65:
        return "Medium"
    else:
        return "High"


# ── Main Batch Predict ────────────────────────────────────────────────────────
def run_batch_predict(input_path: str, output_path: str, batch_size: int = 1000):
    start_time = time.time()

    logger.info("=" * 60)
    logger.info(f"🚀 BATCH PREDICTION STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"   Input  : {input_path}")
    logger.info(f"   Output : {output_path}")
    logger.info("=" * 60)

    # ── 1. Load model ──────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Model not found at '{MODEL_PATH}'. Run model_pipeline.py first.")
        sys.exit(1)

    logger.info(f"📂 Loading model from: {MODEL_PATH}")
    artifact = joblib.load(MODEL_PATH)

    model         = artifact.get("model")
    preprocessor  = artifact.get("preprocessor")
    features      = artifact.get("features")          # filtered feature names
    cat_features  = artifact.get("categorical_features")
    num_features  = artifact.get("numeric_features")

    if model is None:
        logger.error("❌ Loaded artifact does not contain a model key.")
        sys.exit(1)

    logger.info("✅ Model loaded successfully.")

    # ── 2. Load input CSV ──────────────────────────────────────────────────────
    if not os.path.exists(input_path):
        logger.error(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    logger.info(f"📖 Reading input CSV: {input_path}")
    raw_df = pd.read_csv(input_path)
    total_rows = len(raw_df)
    logger.info(f"   Loaded {total_rows:,} records.")

    # ── 3. Feature Engineering ─────────────────────────────────────────────────
    logger.info("🔧 Running feature engineering…")
    df = engineer_features(raw_df)

    # Build model-ready feature df
    X_raw = build_input_df(df, num_features, cat_features)

    # ── 4. Preprocessing (same pipeline as training) ───────────────────────────
    logger.info("⚙  Applying preprocessor (scaling + encoding)…")
    X_enc  = preprocessor.transform(X_raw)
    cat_names    = list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_features))
    encoded_names = num_features + cat_names
    X_df   = pd.DataFrame(X_enc, columns=encoded_names)

    # Apply same correlation-filtered features
    available_features = [f for f in features if f in X_df.columns]
    missing = set(features) - set(available_features)
    if missing:
        logger.warning(f"⚠  {len(missing)} feature(s) missing from input, filling with 0: {missing}")
        for f in missing:
            X_df[f] = 0.0
    X_final = X_df[features].values

    # ── 5. Batch Prediction ────────────────────────────────────────────────────
    logger.info(f"🔮 Running predictions in batches of {batch_size:,}…")
    all_probs  = []
    all_preds  = []

    for start in range(0, len(X_final), batch_size):
        batch = X_final[start: start + batch_size]
        probs = model.predict_proba(batch)[:, 1]
        preds = model.predict(batch)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())

        processed = min(start + batch_size, len(X_final))
        logger.info(f"   Processed {processed:,} / {len(X_final):,} rows…")

    # ── 6. Build Output ────────────────────────────────────────────────────────
    logger.info("📝 Building output dataframe…")

    # Carry over identifier columns if they exist
    id_cols = ["PatientId", "AppointmentID"]
    output_df = pd.DataFrame()
    for col in id_cols:
        if col in raw_df.columns:
            output_df[col] = raw_df[col].values[:len(all_probs)]

    output_df["NoShow_Probability"] = [round(p, 4) for p in all_probs]
    output_df["Prediction"]         = all_preds
    output_df["Label"]              = ["No-Show" if p == 1 else "Will Attend" for p in all_preds]
    output_df["Risk_Level"]         = [assign_risk(p) for p in all_probs]
    output_df["Predicted_At"]       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 7. Save Output ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    output_df.to_csv(output_path, index=False)

    elapsed = time.time() - start_time

    # ── 8. Summary ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"✅ BATCH PREDICTION COMPLETE in {elapsed:.2f}s")
    logger.info(f"   Total records processed : {len(output_df):,}")
    logger.info(f"   Predicted No-Show       : {int(output_df['Prediction'].sum()):,}  "
                f"({output_df['Prediction'].mean()*100:.1f}%)")
    logger.info(f"   Predicted Will Attend   : {int((output_df['Prediction']==0).sum()):,}")
    logger.info(f"   Risk breakdown          : "
                f"Low={int((output_df['Risk_Level']=='Low').sum()):,}, "
                f"Medium={int((output_df['Risk_Level']=='Medium').sum()):,}, "
                f"High={int((output_df['Risk_Level']=='High').sum()):,}")
    logger.info(f"   Output saved to         : {output_path}")
    logger.info("=" * 60)

    return output_df


# ── CLI Entry Point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch predict patient no-show risk from a CSV file."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input CSV file (must have PatientId, Age, Gender, ScheduledDay, AppointmentDay, etc.)"
    )
    parser.add_argument(
        "--output", "-o",
        default="results/batch_predictions.csv",
        help="Path to output CSV file (default: results/batch_predictions.csv)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1000,
        help="Number of rows to process per batch (default: 1000)"
    )

    args = parser.parse_args()
    run_batch_predict(args.input, args.output, args.batch_size)
