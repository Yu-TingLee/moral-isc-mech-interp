import os
import random
import numpy as np
import torch

def seed_everything(seed: int, benchmark: bool = False) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = benchmark

def load_roberta_toxicity_classifier(device: str = "cuda"):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("s-nlp/roberta_toxicity_classifier")
    model = AutoModelForSequenceClassification.from_pretrained(
        "s-nlp/roberta_toxicity_classifier"
    ).to(device)
    model.eval()
    return tokenizer, model


def toxicity_evaluation(texts, roberta_tokenizer, roberta_model):
    if isinstance(texts, str):
        texts = [texts]
    device = next(roberta_model.parameters()).device
    inputs = roberta_tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    with torch.no_grad():
        logits = roberta_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().tolist()
    return probs


def toxicity_evaluation_scalar(texts, roberta_tokenizer, roberta_model):
    full = toxicity_evaluation(texts, roberta_tokenizer, roberta_model)
    return [p[1] for p in full]
