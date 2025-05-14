from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self.model.to(self.device)

    def rerank(self, query: str, docs: list, top_n: int = 4) -> list:
        pairs = [(query, doc) for doc in docs]
        inputs = self.tokenizer(
            [q for q, d in pairs],
            [d for q, d in pairs],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1)
        scored_docs = sorted(
            zip(scores.tolist(), docs), key=lambda x: x[0], reverse=True
        )
        return [{"doc": doc, "score": score} for score, doc in scored_docs[:top_n]]
