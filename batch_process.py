"""
batch_process.py — Daily Automated Model Retraining Pipeline.

This script is triggered automatically by GitHub Actions every day at 00:00 UTC.
It fetches the latest patient data from Supabase, retrains all ML models,
and saves the updated model pipeline and metrics.

Manual run:
    python batch_process.py

With local CSV fallback (if Supabase is unavailable):
    python batch_process.py --local
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("batch_training.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)

# ── Local dataset fallback path ───────────────────────────────────────────────
LOCAL_DATASET = "./Dataset/noshowappointments-kagglev2-may-2016.csv"
MIN_ROWS_REQUIRED = 1000   # Minimum records to proceed with training


def run_batch_training(force_local: bool = False):
    """
    Main entry point for batch model retraining.
    
    Steps:
      1. Attempt to fetch data from Supabase
      2. Fall back to local CSV if Supabase is unavailable or empty
      3. Preprocess the data
      4. Run the model training pipeline
      5. Save updated model + metrics to models/
    """
    start_time = time.time()

    logger.info("=" * 60)
    logger.info(f"🚀 BATCH RETRAINING STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    import model_pipeline  # import here to surface errors cleanly

    df = None

    # ── Step 1: Try Supabase (unless forced local) ────────────────────────────
    if not force_local:
        try:
            from supabase_handler import SupabaseHandler
            handler = SupabaseHandler()
            logger.info("✅ Supabase Handler initialized.")

            table_name = "Patient"
            logger.info(f"📡 Fetching data from Supabase table: '{table_name}'…")
            raw_df = handler.fetch_data(table_name=table_name)

            if raw_df is not None and not raw_df.empty and len(raw_df) >= MIN_ROWS_REQUIRED:
                logger.info(f"✅ Retrieved {len(raw_df):,} records from Supabase.")
                df = model_pipeline.load_and_preprocess_data(raw_df)
            elif raw_df is not None and not raw_df.empty:
                logger.warning(
                    f"⚠  Supabase returned only {len(raw_df)} rows "
                    f"(minimum required: {MIN_ROWS_REQUIRED}). Falling back to local CSV."
                )
            else:
                logger.warning(f"⚠  No data in Supabase table '{table_name}'. Falling back to local CSV.")

        except ValueError as e:
            logger.warning(f"⚠  Supabase credentials not found: {e}. Falling back to local CSV.")
        except Exception as e:
            logger.warning(f"⚠  Supabase connection failed: {e}. Falling back to local CSV.")
    else:
        logger.info("ℹ  --local flag set. Skipping Supabase, using local CSV directly.")

    # ── Step 2: Fall back to local CSV ───────────────────────────────────────
    if df is None:
        if not os.path.exists(LOCAL_DATASET):
            logger.error(f"❌ Local dataset not found at: {LOCAL_DATASET}")
            logger.error("   Place the CSV file at the path above and retry.")
            sys.exit(1)

        logger.info(f"📂 Loading local CSV: {LOCAL_DATASET}")
        df = model_pipeline.load_and_preprocess_data(LOCAL_DATASET)

    # ── Step 3: Validate loaded data ─────────────────────────────────────────
    if df is None or df.empty:
        logger.error("❌ Data loading/preprocessing returned an empty dataset. Aborting.")
        sys.exit(1)

    logger.info(f"📊 Dataset ready: {len(df):,} rows × {len(df.columns)} columns")

    if "Target" not in df.columns:
        logger.error("❌ 'Target' column missing after preprocessing. Check your dataset.")
        sys.exit(1)

    class_dist = df["Target"].value_counts()
    logger.info(f"   Class distribution — Show (0): {class_dist.get(0, 0):,} | "
                f"No-Show (1): {class_dist.get(1, 0):,}")

    # ── Step 4: Train models ──────────────────────────────────────────────────
    logger.info("🛠  Starting model training pipeline…")
    try:
        model_pipeline.main(df)
    except Exception as e:
        import traceback
        logger.error(f"❌ Model training failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    # ── Step 5: Verify output files ───────────────────────────────────────────
    model_file   = os.path.join("models", "best_model_pipeline.joblib")
    metrics_file = os.path.join("models", "metrics.json")

    if os.path.exists(model_file) and os.path.exists(metrics_file):
        model_size_mb = os.path.getsize(model_file) / (1024 * 1024)
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"✨ BATCH RETRAINING COMPLETED SUCCESSFULLY")
        logger.info(f"   Model size  : {model_size_mb:.1f} MB")
        logger.info(f"   Time taken  : {elapsed:.1f}s")
        logger.info(f"   Saved model : {model_file}")
        logger.info(f"   Saved metrics: {metrics_file}")
        logger.info("=" * 60)
    else:
        logger.error("❌ Training completed but output files not found — check model_pipeline.py.")
        sys.exit(1)


# ── CLI Entry Point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Daily batch retraining pipeline for Patient No-Show Prediction Model."
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Force use of local CSV dataset instead of Supabase."
    )
    args = parser.parse_args()
    run_batch_training(force_local=args.local)
