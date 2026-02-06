import ast
import logging
import os
from datetime import datetime

import pandas as pd

from evaluate import evaluate_model
from models import BaselineModel, InstructDeBERTa

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
    # Convert string representation "[('a','Pos'),...]" back to python objects
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

    # 3. Evaluation
    logger.info("Evaluating Proposed Model (Beam=1)...")
    res_v1 = evaluate_model(model_v1, data)
    logger.info(f"Results V1: {res_v1}")

    logger.info("Evaluating Proposed Model (Beam=3)...")
    res_v2 = evaluate_model(model_v2, data)
    logger.info(f"Results V2: {res_v2}")

    logger.info("Evaluating Baseline...")
    res_base = evaluate_model(model_base, data)
    logger.info(f"Results Baseline: {res_base}")

    # 4. Save Results
    results_data = {
        "Model": [
            "Baseline (Sentence-Level)",
            "Instruct-DeBERTa (Beam=1)",
            "Instruct-DeBERTa (Beam=3)",
        ],
        "ATE_F1": ["N/A", res_v1["ATE_F1"], res_v2["ATE_F1"]],
        "ASC_Accuracy": [
            res_base["ASC_Accuracy"],
            res_v1["ASC_Accuracy"],
            res_v2["ASC_Accuracy"],
        ],
        "ASC_F1": [res_base["ASC_F1"], res_v1["ASC_F1"], res_v2["ASC_F1"]],
    }

    df_results = pd.DataFrame(results_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/experiment_results_{timestamp}.csv"

    df_results.to_csv(output_path, index=False)
    logger.info(f"Experiment finished. Results saved to {output_path}")

    print("\n--- FINAL RESULTS TABLE ---")
    print(df_results.to_string(index=False))

    # 5. Demo Output
    print("\n--- DEMO PREDICTION ---")
    demo_text = "The navigation is confusing but the build quality is solid."
    print(f"Input: {demo_text}")
    print(f"Output: {model_v1.predict(demo_text)}")


if __name__ == "__main__":
    main()
