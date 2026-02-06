# Hybrid-ABSA: Instruct-DeBERTa Implementation

This project implements the hybrid ABSA (Aspect-Based Sentiment Analysis) pipeline described in the paper "Instruct-DeBERTa". It combines **InstructABSA** for aspect extraction and **DeBERTa-V3** for sentiment classification.

## Project Structure

* `dataset.csv`: Input data (Restaurant reviews) with ground truth labels.
* `models.py`: Contains the `InstructDeBERTa` class and the `BaselineModel` class.
* `evaluate.py`: Contains logic for calculating F1 scores and Accuracy.
* `main.py`: The entry point script that runs the experiment, logs progress, and saves results.
* `visualize.py`: Generates bar charts comparing the model performances.
* `logs/`: Stores execution logs.
* `results/`: Stores the output CSV files and the generated comparison plots.
* `demo/`: Jupyter Notebooks and related analysis.
* `docs/`: Contains analytical report, slides, and presentation video.

## Setup

1. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2. **Dataset:**
    Ensure `dataset.csv` is in the root directory. (A sample is provided in the project source).

## Execution

1. **Run the Experiment:**

    ```bash
    python main.py
    ```

    This will generate a CSV file in the `results/` folder (e.g., `results/experiment_results_20231027_120000.csv`).

2. **Visualize Results:**
    Pass the generated CSV file to the visualization script:

    ```bash
    python visualize.py --input results/experiment_results_YYYYMMDD_HHMMSS.csv
    ```

    This will save a plot image (e.g., `results/comparison_plot.png`).

## Demo

A Jupyter Notebook is provided for interactive analysis and visualization using a new dataset (Google Colab).

1. Navigate to the demo folder:

    ```bash
    cd demo
    ```

2. Start Jupyter Lab or Notebook:

    ```bash
    jupyter notebook demo.ipynb
    ```

3. Run the cells to load the laptop dataset, run the models, and view the generated performance graphs.

## Experiments

We conducted a comparative study using a subset of the SemEval Restaurant domain to evaluate the robustness of the proposed hybrid architecture against a standard baseline and to analyze the impact of decoding strategies.

### Models Compared

1. **Baseline (Sentence-Level):** * **Model:** `distilbert-base-uncased-finetuned-sst-2-english`.
    * **Description:** A traditional sentiment analysis model that predicts polarity for the entire sentence. It assigns the global sentence sentiment to all aspects found. This serves to demonstrate the limitation of non-ABSA approaches in handling mixed-sentiment sentences (e.g., "Food was good but service was bad").
2. **Instruct-DeBERTa (Proposed, Beam=1):**
    * **ATE:** `InstructABSA` (Tk-Instruct) using Greedy Search.
    * **ASC:** `DeBERTa-V3-ABSA`.
    * **Description:** The primary model proposed in the paper, utilizing efficient greedy decoding for aspect extraction.
3. **Instruct-DeBERTa (Hyperparameter Study, Beam=3):**
    * **Description:** A variation of the proposed model using Beam Search (size=3) during the Aspect Term Extraction phase.
    * **Hypothesis:** Increasing the beam size may improve the recall of extracted aspects in complex sentences, potentially affecting the ATE F1 score.

### Metrics

* **ATE F1:** Harmonic mean of precision and recall for Aspect Term Extraction (Exact Match).
* **ASC Accuracy:** Percentage of correctly classified sentiments for correctly extracted aspects.
* **ASC F1:** Weighted F1 score for sentiment classification.

## References

1. **Paper:** Jayakody, D., et al. "Instruct-DeBERTa: A Hybrid Approach for Aspect-based Sentiment Analysis on Textual Reviews." *arXiv preprint arXiv:2408.13202* (2024).
