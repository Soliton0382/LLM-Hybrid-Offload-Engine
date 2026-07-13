#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sscc_api_client.py -- SSCC as a provider-agnostic API pre-processor.

WHAT IT DOES
------------
Paid LLM APIs (OpenAI, Anthropic, ...) bill you per INPUT token. SSCC compresses
the retrieval context *client-side*, BEFORE the API call, so you pay only for the
salient tokens that actually matter.

    [ raw context + query ]
              |
              v
     SSCC (local, numpy)  -->  keeps the top salient chunks
              |
              v
     paid API (OpenAI / Anthropic)  -->  billed ONLY on the compressed context

Same salience formula as the benchmark:
    score = cos(query, chunk) * (1 + ALPHA * Amp) * (GAMMA + (1 - GAMMA) * Vel)

    cos  : semantic similarity (numpy TF-IDF, zero heavy deps)
    Amp  : amplitude  = local information density of the chunk
    Vel  : velocity   = recency (position bias toward the end of the context)

DEPENDENCIES
------------
    numpy, python-dotenv                     (always)
    openai        (only if LLM_PROVIDER=openai)
    anthropic     (only if LLM_PROVIDER=anthropic)

USAGE
-----
    from sscc_api_client import SSCCClient
    client = SSCCClient()                     # reads .env
    answer = client.ask(query="...", context_chunks=[...])
    print(client.last_stats)                  # tokens_before / after / saved_pct

    # CLI demo (no real API call, shows compression only):
    python sscc_api_client.py --demo
"""
import os, sys, re, math, argparse
from collections import Counter

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except Exception:
    pass


# --------------------------------------------------------------------------- #
#  Configuration (all overridable via .env)                                   #
# --------------------------------------------------------------------------- #
def _f(k, d):  return float(os.getenv(k, d))
def _i(k, d):  return int(os.getenv(k, d))

ALPHA      = _f("SSCC_ALPHA", 0.35)
GAMMA      = _f("SSCC_GAMMA", 0.70)
KEEP_RATIO = _f("SSCC_KEEP_RATIO", 0.35)
MIN_KEEP   = _i("SSCC_MIN_KEEP", 3)
PROVIDER   = os.getenv("LLM_PROVIDER", "openai").lower()


# --------------------------------------------------------------------------- #
#  Pure-numpy TF-IDF embedder (no sklearn, no sentence-transformers)          #
# --------------------------------------------------------------------------- #
_TOKEN = re.compile(r"[a-zA-Z0-9]+")

def _tokenize(text):
    return _TOKEN.findall(text.lower())

class TfidfEmbedder:
    """Minimal TF-IDF vectoriser. Deterministic, dependency-free."""
    def __init__(self, corpus):
        self.vocab = {}
        df = Counter()
        docs = [_tokenize(d) for d in corpus]
        for toks in docs:
            for w in set(toks):
                df[w] += 1
        for w in df:
            self.vocab.setdefault(w, len(self.vocab))
        n = max(1, len(corpus))
        self.idf = np.zeros(len(self.vocab), dtype=np.float32)
        for w, i in self.vocab.items():
            self.idf[i] = math.log((1 + n) / (1 + df[w])) + 1.0

    def embed(self, text):
        v = np.zeros(len(self.vocab), dtype=np.float32)
        toks = _tokenize(text)
        if not toks:
            return v
        tf = Counter(toks)
        for w, c in tf.items():
            j = self.vocab.get(w)
            if j is not None:
                v[j] = (c / len(toks)) * self.idf[j]
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


# --------------------------------------------------------------------------- #
#  Soliton-Salience compressor                                                #
# --------------------------------------------------------------------------- #
class SolitonSalienceCompressor:
    def __init__(self, alpha=ALPHA, gamma=GAMMA):
        self.alpha = alpha
        self.gamma = gamma

    def score(self, query, chunks):
        """Return a salience score per chunk."""
        emb = TfidfEmbedder(chunks + [query])
        q = emb.embed(query)
        n = len(chunks)
        scores = np.zeros(n, dtype=np.float32)
        # Amplitude = information density (unique-token ratio, normalised later)
        amps = np.array([len(set(_tokenize(c))) / max(1, len(_tokenize(c)))
                         for c in chunks], dtype=np.float32)
        amps = amps / (amps.max() + 1e-8)
        for i, c in enumerate(chunks):
            cv = emb.embed(c)
            cos = float(np.dot(q, cv))
            vel = (i + 1) / n                      # recency: later == fresher
            scores[i] = cos * (1 + self.alpha * amps[i]) * \
                        (self.gamma + (1 - self.gamma) * vel)
        return scores

    def compress(self, query, chunks, keep_ratio=KEEP_RATIO, min_keep=MIN_KEEP):
        """Return the salient subset of chunks, preserving original order."""
        if not chunks:
            return []
        scores = self.score(query, chunks)
        k = max(min_keep, int(round(len(chunks) * keep_ratio)))
        k = min(k, len(chunks))
        keep_idx = sorted(np.argsort(-scores)[:k].tolist())
        return [chunks[i] for i in keep_idx]


# --------------------------------------------------------------------------- #
#  Provider adapters                                                          #
# --------------------------------------------------------------------------- #
def _approx_tokens(text):
    """Cheap token estimate (~4 chars/token) for stats when no tokenizer."""
    return max(1, len(text) // 4)

def _call_openai(system, user):
    from openai import OpenAI
    cli = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    r = cli.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.2,
    )
    return r.choices[0].message.content

def _call_anthropic(system, user):
    import anthropic
    cli = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    model = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-20250514")
    r = cli.messages.create(
        model=model, max_tokens=1024, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")

_PROVIDERS = {"openai": _call_openai, "anthropic": _call_anthropic}


# --------------------------------------------------------------------------- #
#  Public client                                                              #
# --------------------------------------------------------------------------- #
class SSCCClient:
    def __init__(self, provider=PROVIDER, keep_ratio=KEEP_RATIO):
        self.provider = provider
        self.keep_ratio = keep_ratio
        self.compressor = SolitonSalienceCompressor()
        self.last_stats = {}

    def ask(self, query, context_chunks, keep_ratio=None, dry_run=False):
        keep = keep_ratio if keep_ratio is not None else self.keep_ratio
        kept = self.compressor.compress(query, context_chunks, keep_ratio=keep)

        raw_ctx = "\n\n".join(context_chunks)
        cmp_ctx = "\n\n".join(kept)
        t_before, t_after = _approx_tokens(raw_ctx), _approx_tokens(cmp_ctx)
        self.last_stats = {
            "chunks_before": len(context_chunks),
            "chunks_after": len(kept),
            "tokens_before": t_before,
            "tokens_after": t_after,
            "saved_pct": round(100 * (1 - t_after / max(1, t_before)), 1),
            "provider": self.provider,
        }

        system = ("You are a precise assistant. Answer ONLY using the context "
                  "below. If the answer is not present, say so.")
        user = f"# Context\n{cmp_ctx}\n\n# Question\n{query}"

        if dry_run:
            return None
        fn = _PROVIDERS.get(self.provider)
        if fn is None:
            raise ValueError(f"Unknown provider '{self.provider}'. "
                             f"Set LLM_PROVIDER to one of: {list(_PROVIDERS)}")
        return fn(system, user)


# --------------------------------------------------------------------------- #
#  CLI demo                                                                    #
# --------------------------------------------------------------------------- #
def _demo():
    secret = "The activation code for reactor B is ZX-4471-DELTA."
    distractors = [f"Log line {i}: routine telemetry, nominal pressure, no anomaly."
                   for i in range(30)]
    chunks = distractors[:15] + [secret] + distractors[15:]
    query = "What is the activation code for reactor B?"

    client = SSCCClient()
    client.ask(query, chunks, dry_run=True)          # compression only, no API
    s = client.last_stats
    print("SSCC compression demo (no API call)")
    print(f"  chunks : {s['chunks_before']} -> {s['chunks_after']}")
    print(f"  tokens : {s['tokens_before']} -> {s['tokens_after']}  "
          f"({s['saved_pct']}% saved)")
    kept = client.compressor.compress(query, chunks, keep_ratio=KEEP_RATIO)
    print(f"  secret kept in compressed context: "
          f"{'YES' if secret in kept else 'NO'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="run compression demo")
    a = ap.parse_args()
    if a.demo or len(sys.argv) == 1:
        _demo()
