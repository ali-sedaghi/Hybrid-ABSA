import os
import ast
import logging
import pandas as pd
from datetime import datetime
from models import InstructDeBERTa, BaselineModel
from evaluate import evaluate_model

# --- Setup Directories ---
os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)

# --- Setup Logging ---
log_filename = f"logs/execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_dataset(path):
    logger.info(f"Loading dataset from {path}...")
    df = pd.read_csv(path)
    dataset = []
    for _, row in df.iterrows():
        try:
            gt = ast.literal_eval(row["ground_truth"])
            dataset.append({"text": row["text"], "ground_truth": gt})
        except Exception as e:
            logger.error(f"Error parsing row: {row['text']} - {e}")
    return dataset


def main():
    logger.info("--- STARTING EXPERIMENT ---")

    # 1. Load Data
    try:
        data = load_dataset("dataset.csv")
        logger.info(f"Loaded {len(data)} examples.")
    except FileNotFoundError:
        logger.error("dataset.csv not found!")
        return

    # 2. Initialize Models
    logger.info("Initializing Instruct-DeBERTa (Beam Size = 1)...")
    model_v1 = InstructDeBERTa(beam_size=1)

    logger.info(
        "Initializing Instruct-DeBERTa (Beam Size = 3) [Hyperparameter Study]..."
    )
    model_v2 = InstructDeBERTa(beam_size=3)

    logger.info("Initializing Baseline (DistilBERT)...")
    model_base = BaselineModel()

    # List to store all detailed rows for final CSV
    all_detailed_records = []

    # 3. Evaluation
    logger.info("Evaluating Proposed Model (Beam=1)...")
    metrics_v1, details_v1 = evaluate_model(
        model_v1, data, model_name="Instruct-DeBERTa (B=1)"
    )
    logger.info(f"Results V1: {metrics_v1}")
    all_detailed_records.extend(details_v1)

    logger.info("Evaluating Proposed Model (Beam=3)...")
    metrics_v2, details_v2 = evaluate_model(
        model_v2, data, model_name="Instruct-DeBERTa (B=3)"
    )
    logger.info(f"Results V2: {metrics_v2}")
    all_detailed_records.extend(details_v2)

    logger.info("Evaluating Baseline...")
    metrics_base, details_base = evaluate_model(model_base, data, model_name="Baseline")
    logger.info(f"Results Baseline: {metrics_base}")
    all_detailed_records.extend(details_base)

    # 4. Save Aggregate Results (Summary Table)
    summary_data = {
        "Model": [
            "Baseline (Sentence-Level)",
            "Instruct-DeBERTa (Beam=1)",
            "Instruct-DeBERTa (Beam=3)",
        ],
        "ATE_F1": ["N/A", metrics_v1["ATE_F1"], metrics_v2["ATE_F1"]],
        "ASC_Accuracy": [
            metrics_base["ASC_Accuracy"],
            metrics_v1["ASC_Accuracy"],
            metrics_v2["ASC_Accuracy"],
        ],
        "ASC_F1": [metrics_base["ASC_F1"], metrics_v1["ASC_F1"], metrics_v2["ASC_F1"]],
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save Summary
    df_summary = pd.DataFrame(summary_data)
    summary_path = f"results/summary_metrics_{timestamp}.csv"
    df_summary.to_csv(summary_path, index=False)

    # Save Detailed Predictions
    df_details = pd.DataFrame(all_detailed_records)
    details_path = f"results/detailed_predictions_{timestamp}.csv"
    df_details.to_csv(details_path, index=False)

    logger.info(f"Summary metrics saved to {summary_path}")
    logger.info(f"Detailed predictions saved to {details_path}")

    print("\n--- FINAL RESULTS TABLE ---")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
