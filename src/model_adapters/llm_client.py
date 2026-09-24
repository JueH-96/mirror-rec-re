"""OpenAI-compatible LLM client with mandatory disk cache and §8.4 call logging.

Cache key: sha256(model_id + prompt + str(temperature) + str(seed)). Every call
(cached or fresh) is appended to model_runs.jsonl with the full §8.4 field set.
Parse failures are retried once (fixed policy, §8.2) with doubled max_tokens,
then recorded as format errors — never repaired by hand.
"""
import hashlib
import json
import re
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "experiments" / "llm_cache"
RUNS_LOG = ROOT / "results" / "model_runs.jsonl"

JSON_RE = re.compile(r"\{[^{}]*\"ranking\"\s*:\s*\[.*?\]\s*\}", re.S)


def cache_key(model_id, prompt, temperature, seed):
    h = hashlib.sha256()
    h.update(model_id.encode())
    h.update(prompt.encode())
    h.update(str(temperature).encode())
    h.update(str(seed).encode())
    return h.hexdigest()


def parse_ranking(raw_text, valid_ids):
    """Extract and validate the ranking JSON (§8.2 checks). Returns (parsed, status)."""
    m = None
    for m in JSON_RE.finditer(raw_text or ""):
        pass  # keep the LAST JSON object (explain-then-rank puts it at the end)
    if m is None:
        return None, "no_json"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, "invalid_json"
    ranking = obj.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        return None, "no_ranking_field"
    ids, scores = [], []
    for entry in ranking:
        if not isinstance(entry, dict) or "item_id" not in entry or "score" not in entry:
            return None, "bad_entry"
        try:
            scores.append(float(entry["score"]))
        except (TypeError, ValueError):
            return None, "unparseable_score"
        ids.append(str(entry["item_id"]))
    if len(set(ids)) != len(ids):
        return None, "duplicate_items"
    if set(ids) - set(valid_ids):
        return None, "foreign_item"
    if set(valid_ids) - set(ids):
        return None, "missing_items"
    return {"ids": ids, "scores": scores}, "ok"


class LLMClient:
    def __init__(self, model_key, model_cfg, temperature=0.0, seed=42, max_retries=1):
        self.model_key = model_key
        self.temperature = temperature
        self.seed = seed
        self.max_retries = max_retries
        api_key = "EMPTY"
        if model_cfg.get("provider") == "deepseek":
            import os
            api_key = os.environ[model_cfg["api_key_env"]]
            self.served_model = "deepseek-chat"
        else:
            self.served_model = model_cfg["hf_repo"]
        self.client = OpenAI(base_url=model_cfg["base_url"], api_key=api_key, timeout=180)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)

    def _call_api(self, prompt, max_tokens):
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.served_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            seed=self.seed,
            max_tokens=max_tokens,
        )
        latency = time.time() - t0
        text = resp.choices[0].message.content
        usage = resp.usage
        return text, latency, {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        }

    def rank(self, request, valid_ids, max_tokens=1024):
        """Returns dict with parsed ranking (or None), parse_status, cached flag."""
        key = cache_key(self.model_key, request["prompt"], self.temperature, self.seed)
        cache_file = CACHE_DIR / f"{key}.json"
        if cache_file.exists():
            cached = json.loads(cache_file.read_text())
            parsed, status = parse_ranking(cached["raw_response"], valid_ids)
            self._log(request, cached["raw_response"], status, cached.get("latency"),
                      cached.get("tokens"), cached=True, attempt=cached.get("attempt", 0))
            return {"parsed": parsed, "parse_status": status, "cached": True,
                    "raw_response": cached["raw_response"]}

        attempt = 0
        text, latency, tokens = self._call_api(request["prompt"], max_tokens)
        parsed, status = parse_ranking(text, valid_ids)
        # Fixed retry policy (§8.2): one retry with doubled budget on parse failure
        # (covers truncation); temperature stays 0 so semantic retries are pointless.
        while status != "ok" and attempt < self.max_retries:
            attempt += 1
            text, latency, tokens = self._call_api(request["prompt"], max_tokens * 2)
            parsed, status = parse_ranking(text, valid_ids)
        cache_file.write_text(json.dumps({
            "raw_response": text, "latency": latency, "tokens": tokens, "attempt": attempt,
        }))
        self._log(request, text, status, latency, tokens, cached=False, attempt=attempt)
        return {"parsed": parsed, "parse_status": status, "cached": False, "raw_response": text}

    def _log(self, request, raw, status, latency, tokens, cached, attempt):
        rec = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model": self.model_key,
            "served_model": self.served_model,
            "prompt_version": request.get("prompt_version"),
            "prompt_variant": request.get("prompt_variant"),
            "request_id": request.get("request_id"),
            "temperature": self.temperature,
            "seed": self.seed,
            "raw_response": raw,
            "parse_status": status,
            "tokens": tokens,
            "latency_s": latency,
            "cost_usd": 0.0,  # local vLLM serving; API models fill this in
            "cached": cached,
            "retry_attempt": attempt,
        }
        with open(RUNS_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
