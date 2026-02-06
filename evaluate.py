from sklearn.metrics import f1_score, accuracy_score


def evaluate_model(model, dataset, model_name="Model"):
    """
    Evaluates the model on the dataset and returns both metrics and detailed prediction logs.
    """
    tp_ate, fp_ate, fn_ate = 0, 0, 0
    y_true_sent = []
    y_pred_sent = []

    # Store per-example details for CSV logging
    detailed_results = []

    is_baseline = hasattr(model, "predict_sentiment")

    for i, entry in enumerate(dataset):
        text = entry["text"]
        gt_pairs = entry["ground_truth"]

        # Normalize Ground Truth: lowercase and strip
        gt_aspects_list = [p[0].lower().strip() for p in gt_pairs]

        # --- PREDICTION ---
        if is_baseline:
            sent_label = model.predict_sentiment(text)
            # Baseline applies sentence sentiment to all valid aspects
            preds = [(a, sent_label) for a in gt_aspects_list]
            raw_output = sent_label  # For logging purpose
        else:
            preds = model.predict(text)
            raw_output = str(preds)

        # --- LOGGING ---
        # Normalize Predictions for consistent logging
        pred_aspects_list = [p[0].lower().strip() for p in preds]

        detailed_results.append(
            {
                "Model": model_name,
                "Input Text": text,
                "Ground Truth": str(gt_pairs),
                "Predicted Raw": raw_output,
                "Extracted Aspects": str(pred_aspects_list),
                "Predicted Sentiments": str([p[1] for p in preds]),
            }
        )

        # --- METRICS CALCULATION (Same as before) ---
        if not is_baseline:
            gt_set = set(gt_aspects_list)
            pred_set = set(pred_aspects_list)

            tp_ate += len(gt_set & pred_set)
            fp_ate += len(pred_set - gt_set)
            fn_ate += len(gt_set - pred_set)

        # Evaluate ASC
        pred_dict = {p[0].lower().strip(): p[1] for p in preds}

        for gt_asp, gt_pol in gt_pairs:
            gt_asp_clean = gt_asp.lower().strip()
            if gt_asp_clean in pred_dict:
                y_true_sent.append(gt_pol)
                y_pred_sent.append(pred_dict[gt_asp_clean])

    # Calculate Aggregate Metrics
    if is_baseline:
        ate_f1 = 0.0
    else:
        precision_ate = tp_ate / (tp_ate + fp_ate) if (tp_ate + fp_ate) > 0 else 0
        recall_ate = tp_ate / (tp_ate + fn_ate) if (tp_ate + fn_ate) > 0 else 0
        ate_f1 = (
            2 * (precision_ate * recall_ate) / (precision_ate + recall_ate)
            if (precision_ate + recall_ate) > 0
            else 0
        )

    asc_acc = accuracy_score(y_true_sent, y_pred_sent) if y_true_sent else 0
    asc_f1 = (
        f1_score(y_true_sent, y_pred_sent, average="weighted") if y_true_sent else 0
    )

    metrics = {
        "ATE_F1": round(ate_f1, 4),
        "ASC_Accuracy": round(asc_acc, 4),
        "ASC_F1": round(asc_f1, 4),
    }

    return metrics, detailed_results
