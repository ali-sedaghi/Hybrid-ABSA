from sklearn.metrics import accuracy_score, f1_score


def evaluate_model(model, dataset, logger=None):
    """
    Evaluates the model on the dataset.
    dataset: List of dictionaries with 'text' and 'ground_truth'
    """
    tp_ate, fp_ate, fn_ate = 0, 0, 0
    y_true_sent = []
    y_pred_sent = []

    # Check if model is the baseline
    is_baseline = hasattr(model, "predict_sentiment")

    for entry in dataset:
        text = entry["text"]
        # Ground truth is expected to be a list of tuples: [('aspect', 'Sentiment'), ...]
        gt_pairs = entry["ground_truth"]
        gt_aspects_list = [p[0] for p in gt_pairs]

        if is_baseline:
            # Baseline: Predict sentence sentiment, apply to ALL ground truth aspects
            sent_label = model.predict_sentiment(text)
            preds = [(a, sent_label) for a in gt_aspects_list]
        else:
            # Instruct-DeBERTa: Full pipeline extraction and classification
            preds = model.predict(text)

        # --- Evaluate ATE (Aspect Term Extraction) ---
        if not is_baseline:
            pred_aspects_list = [p[0] for p in preds]

            gt_set = set(gt_aspects_list)
            pred_set = set(pred_aspects_list)

            tp_ate += len(gt_set & pred_set)
            fp_ate += len(pred_set - gt_set)
            fn_ate += len(gt_set - pred_set)

        # --- Evaluate ASC (Aspect Sentiment Classification) ---
        pred_dict = {p[0]: p[1] for p in preds}

        for gt_asp, gt_pol in gt_pairs:
            # Only evaluate sentiment if the aspect was actually found (or provided in baseline)
            if gt_asp in pred_dict:
                y_true_sent.append(gt_pol)
                y_pred_sent.append(pred_dict[gt_asp])

    # Calculate Metrics
    if is_baseline:
        ate_f1 = 0.0  # Not applicable for baseline
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

    return {
        "ATE_F1": round(ate_f1, 4),
        "ASC_Accuracy": round(asc_acc, 4),
        "ASC_F1": round(asc_f1, 4),
    }
