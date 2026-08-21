"""Scorers, the evaluation suite, and residual-norm calibration."""

import json
import math
import os
import re
import sys

import torch

from src.paths import residual_norms_dir
from src.utils import load_roberta_toxicity_classifier, toxicity_evaluation_scalar

class ToxicityScorer:
    """RoBERTa (s-nlp) + Detoxify."""

    stats_keys = ("roberta_tox", "detoxify_tox")

    def __init__(self, device: str = "cuda"):
        from detoxify import Detoxify

        from src.utils import load_roberta_toxicity_classifier

        self._roberta_tok, self._roberta = load_roberta_toxicity_classifier(device)
        self._detoxify = Detoxify("original", device=device)

    def score(self, texts, prompts=None):
        from src.utils import toxicity_evaluation

        roberta = toxicity_evaluation(texts, self._roberta_tok, self._roberta)
        detox = self._detoxify.predict(list(texts))["toxicity"]
        return {
            "toxicity_probs_RoBERTa": roberta,
            "roberta_tox": [float(r[1]) for r in roberta],
            "detoxify_tox": [float(x) for x in detox],
        }


def rougeL(candidate: str, reference: str) -> float:
    """ROUGE-L F1, the Moral Stories generation metric, as a bare function."""
    from rouge_score import rouge_scorer

    global _ROUGE
    try:
        scorer = _ROUGE
    except NameError:
        scorer = _ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(scorer.score(reference, candidate)["rougeL"].fmeasure)


class MoralScorer:
    """Moral acceptability, via Emelin et al.'s own `action_cls` baseline."""

    stats_keys = ("moral_score",)

    def __init__(self, device: str = "cuda"):
        import pandas as pd
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from src.paths import MORAL_DIR, SCORER_DIR

        ckpt = os.path.join(SCORER_DIR, "moral_roberta")
        if not os.path.isdir(ckpt):
            raise FileNotFoundError(
                f"no moral classifier at {ckpt}. Run train_moral_scorer first, "
                f"and read its gate table before trusting any number it produces."
            )
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(ckpt)
        self._model = AutoModelForSequenceClassification.from_pretrained(ckpt)
        self._model = self._model.to(device).eval()
        self._device = device

        path = os.path.join(MORAL_DIR, "references.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no reference file at {path}. Run python -m experiments.pipeline prepare_moral first."
            )
        df = pd.read_json(path, lines=True)
        self._refs = {
            row["prefix"]: (row["moral_action"], row["immoral_action"])
            for _, row in df.iterrows()
        }

    def _classify(self, prompts, texts, batch_size=32):
        """p(normative) per item."""
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self._tok(
                list(prompts[i:i + batch_size]), list(texts[i:i + batch_size]),
                truncation=True, padding="max_length", max_length=100, return_tensors="pt",
            ).to(self._device)
            with self._torch.no_grad():
                logits = self._model(**enc).logits
            out += self._torch.softmax(logits, dim=-1)[:, 1].tolist()
        return [float(p) for p in out]

    def score(self, texts, prompts=None):
        import sacrebleu

        if prompts is None or len(prompts) != len(texts):
            raise ValueError(
                "MoralScorer needs one prompt per text -- the classifier is fed the "
                "(grounding, target) pair and the reference metrics are keyed by the eval "
                "prefix. Pass prompts= from the caller."
            )

        out = {"moral_score": self._classify(list(prompts), list(texts))}
        for k in ("rougeL_moral", "rougeL_immoral", "bleu_moral", "bleu_immoral"):
            out[k] = []
        for text, prompt in zip(texts, prompts):
            if prompt not in self._refs:
                raise KeyError(f"no references for eval prefix: {prompt[:80]!r}")
            moral, immoral = self._refs[prompt]
            out["rougeL_moral"].append(rougeL(text, moral))
            out["rougeL_immoral"].append(rougeL(text, immoral))
            out["bleu_moral"].append(sacrebleu.sentence_bleu(text, [moral]).score / 100.0)
            out["bleu_immoral"].append(sacrebleu.sentence_bleu(text, [immoral]).score / 100.0)
        return out


SCORERS = {"toxicity": ToxicityScorer, "moral": MoralScorer}


INJECT_KEYS = {
    "toxicity": ("tox", "detoxify"),
    "moral": ("moral", None),
}

_INJECT_SOURCE = {
    "toxicity": ("roberta_tox", "detoxify_tox"),
    "moral": ("moral_score", None),
}


def read_injection_scores(attribute: str, rec: dict, prefix: str = "") -> dict:
    """The inverse of `InjectionScorer.store`: pull an item's scores back out of a stored json as {metric_column: value}, ready to hand to score_generations's `make_row`."""
    suffixes = INJECT_KEYS[attribute]
    columns = _INJECT_SOURCE[attribute]
    out = {}
    for suffix, column in zip(suffixes, columns):
        if column is None or suffix is None:
            continue
        key = InjectionScorer._join(prefix, suffix)
        if key in rec:
            out[column] = rec[key]
    return out


class InjectionScorer:
    """Adapts a SCORERS entry to the (primary, secondary) pair the intervention stages store."""

    def __init__(self, attribute: str, device: str = "cuda"):
        if attribute not in SCORERS:
            raise KeyError(f"no scorer for attribute {attribute!r}. Known: {sorted(SCORERS)}")
        self.attribute = attribute
        self.primary, self.secondary = INJECT_KEYS[attribute]
        self._src_primary, self._src_secondary = _INJECT_SOURCE[attribute]
        self._scorer = SCORERS[attribute](device=device)

    def __call__(self, texts, prompts=None):
        out = self._scorer.score(list(texts), prompts=prompts) \
            if self.attribute == "moral" else self._scorer.score(list(texts))
        first = [float(x) for x in out[self._src_primary]]
        second = [float(x) for x in out[self._src_secondary]] if self._src_secondary else None
        return first, second

    def store(self, record: dict, prefix: str, i: int, primary, secondary):
        """Write one item's scores under `{prefix}_{suffix}`, skipping an absent secondary."""
        record[self.key(prefix)] = primary[i]
        if self.secondary is not None and secondary is not None:
            record[self._join(prefix, self.secondary)] = secondary[i]

    def key(self, prefix: str) -> str:
        """The primary score key a resume check should test for."""
        return self._join(prefix, self.primary)

    @staticmethod
    def _join(prefix: str, suffix: str) -> str:
        return f"{prefix}_{suffix}" if prefix else suffix


JUDGE_RUBRIC = os.path.join("prompts", "judge_coherence_v1.txt")

PPL_MODEL = "gpt2-large"
JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BERTSCORE_MODEL = "roberta-large"
COLA_MODEL = "cointegrated/roberta-large-cola-krishna2020"
METRIC_GROUPS = ("tox", "ppl", "judge", "sim", "distinct2", "cola",
                 "moral")


def distinct2(text: str) -> float:
    """Unique word bigrams / total word bigrams."""
    toks = text.split()
    if len(toks) < 2:
        return float("nan")
    grams = list(zip(toks[:-1], toks[1:]))
    return len(set(grams)) / len(grams)


class EvalSuite:
    def __init__(
        self,
        device: str = "cuda",
        ppl_model: str = PPL_MODEL,
        judge_model: str = JUDGE_MODEL,
        bertscore_model: str = BERTSCORE_MODEL,
        cola_model: str = COLA_MODEL,
        hf_token: str = None,
        max_ppl_tokens: int = 1024,
        rescale_bertscore: bool = True,
    ):
        self.device = device
        self.ppl_model_name = ppl_model
        self.judge_model_name = judge_model
        self.bertscore_model_name = bertscore_model
        self.cola_model_name = cola_model
        self.hf_token = hf_token or os.environ.get("HF_TOKEN") or None
        self.max_ppl_tokens = max_ppl_tokens
        self.rescale_bertscore = rescale_bertscore

        self._roberta = None
        self._detoxify = None
        self._ppl = None
        self._judge = None
        self._bertscorer = None
        self._cola = None

        self.judge_parse_failures = 0
        self.judge_calls = 0

    @property
    def roberta(self):
        if self._roberta is None:
            self._roberta = load_roberta_toxicity_classifier(self.device)
        return self._roberta

    @property
    def detoxify(self):
        if self._detoxify is None:
            from detoxify import Detoxify

            self._detoxify = Detoxify("original", device=self.device)
        return self._detoxify

    @property
    def ppl(self):
        if self._ppl is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.ppl_model_name)
            tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                self.ppl_model_name, dtype=torch.float32
            ).to(self.device)
            model.eval()
            self._ppl = (tok, model)
        return self._ppl

    @property
    def judge(self):
        if self._judge is None:
            from src.model import load_model

            model, tok = load_model(
                self.judge_model_name,
                dtype=torch.float16,
                device_map="auto",
                token=self.hf_token,
                padding_side="left",
            )
            self._judge = (tok, model, open(JUDGE_RUBRIC).read())
        return self._judge

    @property
    def bertscorer(self):
        if self._bertscorer is None:
            from bert_score import BERTScorer

            self._bertscorer = BERTScorer(
                model_type=self.bertscore_model_name,
                lang="en",
                rescale_with_baseline=self.rescale_bertscore,
                device=self.device,
            )
        return self._bertscorer

    @property
    def cola(self):
        if self._cola is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.cola_model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.cola_model_name
            ).to(self.device)
            model.eval()
            self._cola = (tok, model)
        return self._cola

    def free(self):
        self._roberta = self._detoxify = self._ppl = self._judge = self._bertscorer = None
        self._cola = None
        torch.cuda.empty_cache()

    def fluency_cola(self, texts, batch_size: int = 32):
        """P(linguistically acceptable) in [0, 1], per text."""
        tok, model = self.cola
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = [t if t.strip() else " " for t in texts[i : i + batch_size]]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=512).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(model(**enc).logits, dim=-1)
            out += probs[:, 0].cpu().tolist()
        return out

    def toxicity(self, texts, batch_size: int = 32):
        """-> (roberta_tox, detoxify_tox), both P(toxic) in [0, 1]."""
        rob, det = [], []
        for i in range(0, len(texts), batch_size):
            chunk = [t if t.strip() else " " for t in texts[i : i + batch_size]]
            rob += toxicity_evaluation_scalar(chunk, *self.roberta)
            det += [float(x) for x in self.detoxify.predict(chunk)["toxicity"]]
        return rob, det

    def perplexity(self, prefixes, continuations, batch_size: int = 8):
        """PPL of each continuation under gpt2-large, with its prefix in context."""
        tok, model = self.ppl
        out = [float("nan")] * len(prefixes)

        encoded = []
        for i, (pre, cont) in enumerate(zip(prefixes, continuations)):
            if not cont.strip():
                continue
            if not cont[0].isspace():
                cont = " " + cont
            pre_ids = tok(pre)["input_ids"]
            cont_ids = tok(cont)["input_ids"]
            if not cont_ids:
                continue
            ids = (pre_ids + cont_ids)[-self.max_ppl_tokens :]
            n_cont = min(len(cont_ids), len(ids) - 1)
            if n_cont < 1:
                continue
            encoded.append((i, ids, n_cont))

        encoded.sort(key=lambda e: len(e[1]))
        for start in range(0, len(encoded), batch_size):
            batch = encoded[start : start + batch_size]
            width = max(len(ids) for _, ids, _ in batch)

            input_ids = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long)
            attn = torch.zeros((len(batch), width), dtype=torch.long)
            labels = torch.full((len(batch), width), -100, dtype=torch.long)
            for b, (_, ids, n_cont) in enumerate(batch):
                input_ids[b, : len(ids)] = torch.tensor(ids)
                attn[b, : len(ids)] = 1
                labels[b, len(ids) - n_cont : len(ids)] = torch.tensor(ids[-n_cont:])

            input_ids, attn, labels = (t.to(self.device) for t in (input_ids, attn, labels))
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attn).logits

            shift_logits = logits[:, :-1].float()
            shift_labels = labels[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                shift_logits.transpose(1, 2), shift_labels, reduction="none", ignore_index=-100
            )
            counts = (shift_labels != -100).sum(dim=1).clamp(min=1)
            mean_nll = loss.sum(dim=1) / counts

            for b, (i, _, _) in enumerate(batch):
                out[i] = float(math.exp(min(float(mean_nll[b]), 20.0)))
        return out

    def judge_fluency(self, prefixes, continuations, batch_size: int = 8, max_chars: int = 2000):
        """Llama-3.1-8B-Instruct coherence, 0-10."""
        tok, model, rubric = self.judge
        out = [float("nan")] * len(prefixes)

        chats = [
            tok.apply_chat_template(
                [{"role": "user", "content": rubric.format(
                    prompt=pre.strip() or "(empty)",
                    continuation=cont[:max_chars].strip() or "(empty)",
                )}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for pre, cont in zip(prefixes, continuations)
        ]

        for start in range(0, len(chats), batch_size):
            batch = chats[start : start + batch_size]
            inputs = tok(batch, padding="longest", return_tensors="pt").to(model.device)
            with torch.no_grad():
                seqs = model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            replies = tok.batch_decode(
                seqs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            for b, reply in enumerate(replies):
                self.judge_calls += 1
                score = _parse_score(reply)
                if score is None:
                    self.judge_parse_failures += 1
                else:
                    out[start + b] = score
        return out

    def bertscore_f1(self, candidates, references, batch_size: int = 32):
        """F1 of each candidate against its reference."""
        out = [float("nan")] * len(candidates)
        keep = [
            i for i, (c, r) in enumerate(zip(candidates, references))
            if c.strip() and r.strip()
        ]
        if not keep:
            return out
        _, _, f1 = self.bertscorer.score(
            [candidates[i] for i in keep],
            [references[i] for i in keep],
            batch_size=batch_size,
        )
        for j, i in enumerate(keep):
            out[i] = float(f1[j])
        return out

_SCORE_RE = re.compile(r"\b(10|[0-9])\b")


def _parse_score(reply: str):
    """First integer 0-10 in the reply."""
    m = _SCORE_RE.search(reply)
    if not m:
        return None
    v = float(m.group(1))
    return v if 0.0 <= v <= 10.0 else None


def score_rows(suite: EvalSuite, rows, metrics=METRIC_GROUPS, batch_size: int = 8):
    """Fill the metric columns of tidy rows in place."""
    texts = [r["text"] for r in rows]
    prefixes = [r["_prefix"] for r in rows]
    refs = [r["_reference"] for r in rows]

    if "tox" in metrics:
        rob, det = suite.toxicity(texts)
        for r, a, b in zip(rows, rob, det):
            r["roberta_tox"], r["detoxify_tox"] = a, b
    if "ppl" in metrics:
        for r, v in zip(rows, suite.perplexity(prefixes, texts, batch_size=batch_size)):
            r["ppl"] = v
    if "sim" in metrics:
        for r, v in zip(rows, suite.bertscore_f1(texts, refs)):
            r["bertscore_f1"] = v
    if "distinct2" in metrics:
        for r in rows:
            r["distinct2"] = distinct2(r["text"])
    if "judge" in metrics:
        for r, v in zip(rows, suite.judge_fluency(prefixes, texts, batch_size=batch_size)):
            r["judge_fluency"] = v
    if "cola" in metrics:
        for r, v in zip(rows, suite.fluency_cola(texts)):
            r["cola_fluency"] = v
    return rows


def load_resid_norms(model_name: str, n_layers: int, attribute: str = "toxicity") -> dict:
    """median ||x^(l)|| at the injection site (the input of block l)."""
    path = os.path.join(residual_norms_dir(model_name, attribute), "diagnostics.json")
    if not os.path.exists(path):
        flag = "" if attribute == "toxicity" else f" --attribute {attribute}"
        sys.exit(
            f"missing {path}\n"
            f"Coefficients are scaled by the residual norm, so measure_residual_norms must run first:\n"
            f"  python -m experiments.measure residual_norms --model {model_name}{flag}"
        )

    norms = {int(r["layer"]): float(r["norm_in"]) for r in json.load(open(path))["layers"]}
    missing = [l for l in range(n_layers) if l not in norms]
    if missing:
        sys.exit(f"{path} has no norm for layers {missing}; re-run measure_residual_norms on this model")
    return norms


def resolve_coef(coef: float, layer: int, norms: dict, mode: str) -> float:
    """Turn a swept coefficient into the absolute scalar the hook multiplies by."""
    if mode == "norm":
        return coef * norms[layer]
    if mode == "absolute":
        return coef
    raise ValueError(f"unknown coefficient mode {mode!r} (expected 'norm' or 'absolute')")
