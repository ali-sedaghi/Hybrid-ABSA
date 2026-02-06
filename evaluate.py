from sklearn.metrics import f1_score, accuracy_score


def evaluate_model(model, dataset, logger=None):
    """
    Evaluates the model on the dataset with DEBUG prints.
    """
    tp_ate, fp_ate, fn_ate = 0, 0, 0
    y_true_sent = []
    y_pred_sent = []

    is_baseline = hasattr(model, "predict_sentiment")

    print(
        f"\n--- Starting Evaluation (Model: {'Baseline' if is_baseline else 'Instruct-DeBERTa'}) ---"
    )

    for i, entry in enumerate(dataset):
        text = entry["text"]
        gt_pairs = entry["ground_truth"]

        # Normalize Ground Truth
        gt_aspects_list = [p[0].lower().strip() for p in gt_pairs]

        if is_baseline:
            sent_label = model.predict_sentiment(text)
            preds = [(a, sent_label) for a in gt_aspects_list]
        else:
            preds = model.predict(text)

        # --- Evaluate ATE ---
        if not is_baseline:
            # Normalize Predictions
            pred_aspects_list = [p[0].lower().strip() for p in preds]

            gt_set = set(gt_aspects_list)
            pred_set = set(pred_aspects_list)

            # --- DEBUG COMPARISON ---
            # Print only first 5 examples or if there is a mismatch to avoid flooding logs
            if i < 5 or len(gt_set & pred_set) == 0:
                print(f"\n[DEBUG EVAL] Example {i + 1}:")
                print(f"  GT Aspects  : {gt_set}")
                print(f"  Pred Aspects: {pred_set}")
                if len(gt_set & pred_set) == 0 and len(gt_set) > 0:
                    print("  -> FAILURE: No overlap!")
            # ------------------------

            tp_ate += len(gt_set & pred_set)
            fp_ate += len(pred_set - gt_set)
            fn_ate += len(gt_set - pred_set)

        # --- Evaluate ASC ---
        pred_dict = {p[0].lower().strip(): p[1] for p in preds}

        for gt_asp, gt_pol in gt_pairs:
            gt_asp_clean = gt_asp.lower().strip()
            if gt_asp_clean in pred_dict:
                y_true_sent.append(gt_pol)
                y_pred_sent.append(pred_dict[gt_asp_clean])

    # Calculate Metrics
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

    return {
        "ATE_F1": round(ate_f1, 4),
        "ASC_Accuracy": round(asc_acc, 4),
        "ASC_F1": round(asc_f1, 4),
    }
