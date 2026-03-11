import re
import ast
import math
import json
import faiss
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter
from schemas import EvidenceNode, MCQZipAnswer


# ----------------------------
# TEXT & PARSING UTILS
# ----------------------------
def clean_options_dict(options: Dict[str, Any]) -> Dict[str, str]:
    return {str(k): str(v) for k, v in options.items() if str(k).isdigit() or str(k).lower() in ['a','b','c','d','e']}

def safe_literal_eval_dict(x: Any) -> Dict[str, str]:
    if isinstance(x, dict):
        return {str(k): str(v) for k, v in x.items()}
    if isinstance(x, str):
        s = x.strip()
        try:
            obj = json.loads(s)
            if isinstance(obj, dict): return {str(k): str(v) for k, v in obj.items()}
        except Exception: pass
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, dict): return {str(k): str(v) for k, v in obj.items()}
        except Exception: pass
    return {}

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def safe_sort_key(k):
    k_str = str(k)
    if k_str.isdigit(): return (0, int(k_str))
    return (1, k_str)

def build_joint_query(question: str, options: Dict[str, str]) -> str:
    opt_str = " ".join([f"({k}) {v}" for k, v in sorted(options.items(), key=lambda kv: safe_sort_key(kv[0]))])
    return normalize_text(f"Question: {question}\nOptions: {opt_str}")

def build_question_only_query(question: str) -> str:
    return normalize_text(f"Question: {question}")

def get_nota_key(options: Dict[str, str]) -> Optional[str]:
    """Finds the key for 'None of the above' if it exists."""
    for k, v in options.items():
        if "none of the above" in str(v).lower():
            return str(k)
    return None

EID_RE = re.compile(r"\[(E\d+)\]")

def extract_eids_from_text(*texts: str) -> List[str]:
    found = []
    for t in texts:
        if t: found.extend(EID_RE.findall(t))
    out = []
    seen = set()
    for e in found:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out

# ----------------------------
# MATH & BM25 UTILS
# ----------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))

def minmax01(x: np.ndarray) -> np.ndarray:
    if x.size == 0: return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi <= lo + 1e-12: return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
def bm25_tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]

def bm25_scores(query: str, docs: List[str], k1: float = 1.6, b: float = 0.75) -> np.ndarray:
    if not docs: return np.zeros((0,), dtype=np.float32)
    q_tokens = bm25_tokenize(query)
    if not q_tokens: return np.zeros((len(docs),), dtype=np.float32)

    docs_tokens = [bm25_tokenize(d) for d in docs]
    N = len(docs_tokens)
    dl = np.array([len(toks) for toks in docs_tokens], dtype=np.float32)
    avgdl = max(float(np.mean(dl)), 1.0)

    df = {}
    for toks in docs_tokens:
        for w in set(toks): df[w] = df.get(w, 0) + 1

    def idf(w):
        n = df.get(w, 0)
        return math.log(1.0 + (N - n + 0.5) / (n + 0.5))

    scores = np.zeros((N,), dtype=np.float32)
    for i, toks in enumerate(docs_tokens):
        if not toks: continue
        tf = Counter(toks)
        s = 0.0
        denom_base = k1 * (1.0 - b + b * (dl[i] / avgdl))
        for w in q_tokens:
            f = tf.get(w, 0)
            if f <= 0: continue
            numer = f * (k1 + 1.0)
            denom = f + denom_base
            s += idf(w) * (numer / max(denom, 1e-12))
        scores[i] = float(s)
    return scores

# ----------------------------
# MMR SELECTION
# ----------------------------
def mmr_select(query_emb: np.ndarray, cand_embs: np.ndarray, cand_scores: np.ndarray, k: int = 60, lambda_mult: float = 0.7) -> List[int]:
    if cand_embs.shape[0] == 0: return []
    k = min(k, cand_embs.shape[0])

    q = query_emb.astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-12)

    E = cand_embs.astype(np.float32)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    rel = (E @ q).reshape(-1)
    s_norm = cand_scores.astype(np.float32)
    s_norm = (s_norm - np.min(s_norm)) / (np.max(s_norm) - np.min(s_norm) + 1e-12)
    rel2 = 0.7 * rel + 0.3 * s_norm

    selected = []
    remaining = list(range(E.shape[0]))

    while len(selected) < k and remaining:
        if not selected:
            best = max(remaining, key=lambda i: float(rel2[i]))
            selected.append(best)
            remaining.remove(best)
            continue

        S = E[selected]
        sims = E[remaining] @ S.T
        max_sim = np.max(sims, axis=1)

        mmr = lambda_mult * rel2[remaining] - (1.0 - lambda_mult) * max_sim
        best_pos = int(np.argmax(mmr))
        best = remaining[best_pos]
        selected.append(best)
        remaining.remove(best)

    return selected

# ----------------------------
# PRUNING LOGIC
# ----------------------------
def node_content_for_scoring(n: EvidenceNode) -> str:
    return normalize_text(n.text)

def compute_node_bm25_scores(query: str, nodes: List[EvidenceNode]) -> np.ndarray:
    docs = [node_content_for_scoring(n) for n in nodes]
    return bm25_scores(query, docs)

def mcq_discriminative_margin(question: str, options: Dict[str, str], nodes: List[EvidenceNode]) -> np.ndarray:
    if not nodes or not options: return np.zeros((len(nodes),), dtype=np.float32)
    
    option_keys = sorted(options.keys(), key=safe_sort_key)
    queries = [normalize_text(f"{question} {options[k]}") for k in option_keys]
    docs = [node_content_for_scoring(n) for n in nodes]

    per_opt = []
    for q in queries: per_opt.append(bm25_scores(q, docs))
    M = np.stack(per_opt, axis=1)

    sorted_vals = np.sort(M, axis=1)[:, ::-1]
    best = sorted_vals[:, 0]
    second = sorted_vals[:, 1] if sorted_vals.shape[1] >= 2 else np.zeros_like(best)
    margin = best - second
    return margin.astype(np.float32)

def prune_nodes(mode: str, question: str, options: Dict[str, str], nodes: List[EvidenceNode], keep_k: int, min_keep: int = 4) -> List[EvidenceNode]:
    if not nodes: return []
    mode = (mode or "mcq").lower().strip()
    keep_k = max(int(keep_k), int(min_keep))

    base = np.array([float(n.score01) for n in nodes], dtype=np.float32)
    base = minmax01(base)

    if mode == "chat":
        q = build_question_only_query(question)
        bm25_q = compute_node_bm25_scores(q, nodes)
        bm25_q = minmax01(bm25_q)
        final = 0.55 * base + 0.30 * bm25_q
    else:
        joint = build_joint_query(question, options) if options else build_question_only_query(question)
        bm25_q = compute_node_bm25_scores(joint, nodes)
        bm25_q = minmax01(bm25_q)
        margin = mcq_discriminative_margin(question, options, nodes) if options else np.zeros_like(bm25_q)
        margin = minmax01(margin)
        final = (0.55 * base) + (0.30 * bm25_q) + (0.15 * margin)
        #final = (0.40 * base) + (0.25 * bm25_q) + (0.35 * margin)

    order = np.argsort(-final)
    chosen = order[: min(keep_k, len(nodes))].tolist()
    chosen = chosen[: max(min_keep, len(chosen))]

    out = []
    for i in chosen:
        n = nodes[int(i)]
        out.append(EvidenceNode(n.eid, n.route, float(final[int(i)]), n.text, n.meta, n.trace))
    return out

# ----------------------------
# SANITIZATION (Logic Updated for NOTA)
# ----------------------------
def extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not text: return None
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m: return None
    blob = m.group(0).strip()
    try:
        obj = json.loads(blob)
        # --- FIX: Ensure it is a DICT, not a List ---
        if isinstance(obj, list):
            # If LLM returned a list, maybe the first item is the answer?
            if len(obj) > 0 and isinstance(obj[0], dict):
                return obj[0]
            return {} # Invalid format
        return obj
    except Exception: pass
    
    try:
        obj = ast.literal_eval(blob)
        if isinstance(obj, dict): return obj
    except Exception: return None
    return None

def sanitize_zip_answer(obj: Dict[str, Any], options: Dict[str, str], allowed_eids: List[str], fallback_reason: str, fallback_evidence: Optional[List[str]]) -> MCQZipAnswer:
    
    if isinstance(obj, list):
        # If LLM returned a list, take the first item if it's a dict
        if len(obj) > 0 and isinstance(obj[0], dict):
            obj = obj[0]
        else:
            obj = {}
    
    # If it's still not a dict (e.g. None or string), force it to empty dict
    if not isinstance(obj, dict):
        obj = {}
    
    option_keys = sorted(options.keys(), key=safe_sort_key)
    option_key_set = set(option_keys)
    allowed_set = set(allowed_eids)

    # 1. Detect if "None of the above" exists
    nota_key = get_nota_key(options)

    # 2. Extract Raw Choice from LLM
    cop = str(obj.get("cop_index", "-1"))
    if cop not in option_key_set and cop != "-1": 
        cop = "-1"

    why_correct = str(obj.get("why_correct", "") or "").strip()
    why_others = obj.get("why_others_incorrect", {})
    if not isinstance(why_others, dict): 
        why_others = {}
    why_others = {str(k): str(v) for k, v in why_others.items()}

    raw_why_others = obj.get("why_others_incorrect", {})
    why_others = {}
    
    if isinstance(raw_why_others, dict):
        # Normal case: LLM followed instructions
        why_others = {str(k): str(v) for k, v in raw_why_others.items()}
    elif isinstance(raw_why_others, list):
        # LLM hallucinated a list. Map it to the remaining option keys sequentially.
        remaining_keys = [k for k in option_keys if k != cop]
        for i, val in enumerate(raw_why_others):
            if i < len(remaining_keys):
                why_others[remaining_keys[i]] = str(val)
    elif isinstance(raw_why_others, str):
        # LLM hallucinated a single string.
        why_others = {"all": raw_why_others}
        
    # 3. Extract Citations FIRST to check if the LLM actually used evidence
    cited = extract_eids_from_text(why_correct, *list(why_others.values()))
    evidence_used = [e for e in cited if e in allowed_set]

    # 4. OVERRIDE LOGIC FOR NOTA
    # If the LLM failed to cite any evidence OR if it explicitly abstained (-1),
    # and a "None of the above" option exists, force the answer to NOTA.
    if nota_key and (not evidence_used or cop == "-1"):
        cop = nota_key
        why_correct = "Evidence was insufficient to support any specific option, so 'None of the above' is selected."

    # 5. Set Answer Text based on final COP
    answer = "" if cop == "-1" else options.get(cop, "")

    # 6. Fallback Reason Handling for non-NOTA scenarios (e.g. regular FCT task)
    if not why_correct:
        cop = "-1"
        answer = ""
        why_correct = fallback_reason
        
    # Ensure answer text is empty if we end up at -1
    if cop == "-1": 
        answer = ""

    # 7. Rebuild why_others_incorrect to exclude the final chosen option
    keys_to_include = option_keys[:] if cop == "-1" else [k for k in option_keys if k != cop]
    ordered_why_others = {k: (why_others.get(k, "") or "").strip() for k in keys_to_include}

    return MCQZipAnswer(
        cop_index=cop,
        answer=answer,
        why_correct=why_correct,
        why_others_incorrect=ordered_why_others,
        evidence_used=evidence_used,
        evidence={}
    )

# ----------------------------
# FAISS WRAPPER
# ----------------------------
def read_json_or_jsonl(path: str) -> Any:
    p = str(path)
    if p.endswith(".jsonl"):
        out = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): out.append(json.loads(line))
        return out
    with open(p, "r", encoding="utf-8") as f: return json.load(f)

class FaissIndex:
    def __init__(self, index_path: str, meta_path: str):
        self.index = faiss.read_index(index_path)
        self.meta = read_json_or_jsonl(meta_path)
    def search(self, emb: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        D, I = self.index.search(emb.astype(np.float32), topk)
        return D[0], I[0]