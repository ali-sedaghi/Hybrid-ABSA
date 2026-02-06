import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


class InstructDeBERTa:
    """
    The Proposed Hybrid Model from the paper.
    ATE Component: InstructABSA (Tk-Instruct base)
    ASC Component: DeBERTa-V3-base-absa-v1
    """

    def __init__(self, device=None, beam_size=1):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.beam_size = beam_size
        print(
            f"[Instruct-DeBERTa] Loading on {self.device} (Beam Size: {self.beam_size})..."
        )

        # --- ATE Component (Aspect Term Extraction) ---
        ate_model_id = "kevinscaria/ate_tk-instruct-base-def-pos-neg-neut-combined"
        self.ate_tokenizer = AutoTokenizer.from_pretrained(ate_model_id)
        self.ate_model = AutoModelForSeq2SeqLM.from_pretrained(ate_model_id).to(
            self.device
        )

        # --- ASC Component (Aspect Sentiment Classification) ---
        asc_model_id = "yangheng/deberta-v3-base-absa-v1.1"
        self.asc_pipeline = pipeline(
            "text-classification",
            model=asc_model_id,
            tokenizer=asc_model_id,
            device=0 if self.device == "cuda" else -1,
        )

    def extract_aspects(self, text):
        input_ids = self.ate_tokenizer(text, return_tensors="pt").input_ids.to(
            self.device
        )
        with torch.no_grad():
            outputs = self.ate_model.generate(
                input_ids, max_length=128, num_beams=self.beam_size
            )
        decoded = self.ate_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Parse output: "aspect1, aspect2" -> ["aspect1", "aspect2"]
        aspects = [a.strip() for a in decoded.split(",") if a.strip()]
        return aspects

    def predict(self, text):
        extracted_aspects = self.extract_aspects(text)
        predictions = []

        for aspect in extracted_aspects:
            try:
                # DeBERTa ASC prediction
                res = self.asc_pipeline(text, text_pair=aspect)
                label = res[0]["label"]
                predictions.append((aspect, label))
            except Exception as e:
                continue
        return predictions


class BaselineModel:
    """
    Baseline: Standard Sentence-Level Sentiment Analysis.
    Uses DistilBERT fine-tuned on SST-2.
    """

    def __init__(self, device=None):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[Baseline] Loading DistilBERT on {self.device}...")
        self.pipe = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if self.device == "cuda" else -1,
        )

    def predict_sentiment(self, text):
        res = self.pipe(text)[0]
        label = res["label"]
        if label == "POSITIVE":
            return "Positive"
        if label == "NEGATIVE":
            return "Negative"
        return "Neutral"
