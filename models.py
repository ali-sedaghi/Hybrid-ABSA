import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


class InstructDeBERTa:
    """
    The Proposed Hybrid Model.
    ATE: InstructABSA (Tk-Instruct)
    ASC: DeBERTa-V3-ABSA
    """

    def __init__(self, device=None, beam_size=1):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.beam_size = beam_size
        print(
            f"[Instruct-DeBERTa] Loading on {self.device} (Beam Size: {self.beam_size})..."
        )

        # --- ATE Component ---
        ate_model_id = "kevinscaria/ate_tk-instruct-base-def-pos-neg-neut-combined"
        self.ate_tokenizer = AutoTokenizer.from_pretrained(ate_model_id)
        self.ate_model = AutoModelForSeq2SeqLM.from_pretrained(ate_model_id).to(
            self.device
        )

        # --- ASC Component ---
        asc_model_id = "yangheng/deberta-v3-base-absa-v1.1"
        self.asc_pipeline = pipeline(
            "text-classification",
            model=asc_model_id,
            tokenizer=asc_model_id,
            device=0 if self.device == "cuda" else -1,
        )

    def extract_aspects(self, text):
        # --- FIX: STRICT INSTRUCTION PROMPT ---
        # Tk-Instruct models are sensitive to newlines and "Definition/Input/Output" markers.
        # We explicitly tell it to use commas.
        prompt = (
            "Definition: Extract the aspect terms from the text. Return only the aspect terms separated by commas.\n"
            f"Input: {text}\n"
            "Output:"
        )

        input_ids = self.ate_tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )

        with torch.no_grad():
            outputs = self.ate_model.generate(
                input_ids, max_length=128, num_beams=self.beam_size
            )
        decoded = self.ate_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # --- DEBUG PRINT (Uncomment if issues persist) ---
        # print(f"\n[DEBUG] Raw Output: '{decoded}'")

        # Cleaning: Remove "Aspect terms:" prefix if the model generates it
        cleaned = decoded.replace("Aspect terms:", "").replace("Output:", "").strip()

        # Parse comma-separated list
        aspects = [a.strip() for a in cleaned.split(",") if a.strip()]

        # Safety check: If extracted text is identical to input (copying), discard it
        if len(aspects) == 1 and aspects[0].lower() == text.lower():
            return []

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
