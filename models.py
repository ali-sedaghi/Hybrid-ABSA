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
        # --- FIX: ADD INSTRUCTION PROMPT ---
        # The model requires a definition to know it should perform extraction.
        # Based on InstructABSA (Scaria et al.), the format is:
        prompt = f"Definition: The task is to extract the aspect terms from the given sentence. Sentence: {text}"

        input_ids = self.ate_tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )

        with torch.no_grad():
            outputs = self.ate_model.generate(
                input_ids, max_length=128, num_beams=self.beam_size
            )
        decoded = self.ate_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # --- DEBUG PRINT ---
        # print(f"\n[DEBUG ATE] Input: '{prompt}'")
        # print(f"[DEBUG ATE] Raw Model Output: '{decoded}'")
        # -------------------

        # Parse: "aspect1, aspect2" -> ["aspect1", "aspect2"]
        # The model might output "food, service" or "[food, service]" depending on specific tuning
        cleaned_output = decoded.replace("[", "").replace("]", "")
        aspects = [a.strip() for a in cleaned_output.split(",") if a.strip()]

        return aspects

    def predict(self, text):
        extracted_aspects = self.extract_aspects(text)
        predictions = []

        if not extracted_aspects:
            print(f"[DEBUG ATE] No aspects extracted for: '{text}'")

        for aspect in extracted_aspects:
            try:
                res = self.asc_pipeline(text, text_pair=aspect)
                label = res[0]["label"]
                predictions.append((aspect, label))
            except Exception as e:
                print(f"[DEBUG ASC] Error classifying '{aspect}': {e}")
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
