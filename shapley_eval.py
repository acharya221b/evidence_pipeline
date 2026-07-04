import os
import json
import logging
import argparse
import asyncio
import itertools
import math
import pandas as pd
from tqdm import tqdm
import traceback
from typing import List, Dict, Any

from utils import safe_literal_eval_dict
from schemas import EvidenceNode
from pipeline import AsyncKGMCQPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger("shapley_eval")

def dict_to_node(d: Dict[str, Any]) -> EvidenceNode:
    meta = d.get('meta', {})
    if not meta:
        if 'ATUI' in d: meta['ATUI'] = d['ATUI']
        if 'SAB' in d: meta['SAB'] = d['SAB']
        if 'sui' in d: meta['sui'] = d['sui']
        if 'name' in d: meta['name'] = d['name']
    return EvidenceNode(
        eid=d['eid'], route=d['route'], score01=d['score'],
        text=d['text'], meta=meta, trace=d.get('trace')
    )

def calculate_shapley_values(nodes: List[EvidenceNode], subset_results: Dict[tuple, int]) -> Dict[str, float]:
    N = len(nodes)
    node_eids = [n.eid for n in nodes]
    shapley_values = {eid: 0.0 for eid in node_eids}
    
    for i, eid in enumerate(node_eids):
        subsets_without_i = [s for s in subset_results.keys() if eid not in s]
        for S in subsets_without_i:
            S_len = len(S)
            weight = (math.factorial(S_len) * math.factorial(N - S_len - 1)) / math.factorial(N)
            v_S = subset_results[S]
            S_with_i = tuple(sorted(list(S) + [eid]))
            v_S_i = subset_results[S_with_i]
            marginal_contribution = v_S_i - v_S
            shapley_values[eid] += weight * marginal_contribution
            
    return shapley_values

async def process_shapley_row(item: Dict[str, Any], pipeline: AsyncKGMCQPipeline, sem: asyncio.Semaphore, pbar: tqdm, task_name: str) -> Dict[str, Any]:
    async with sem:
        try:
            q_id = item['id']
            question = item['question']
            options = item['options']
            correct_idx = str(item['correct_index']).strip()
            
            all_nodes = [dict_to_node(n) for n in item['all_evidence']]
            original_cited = set(item['evidence_used'])
            
            # --- THE "FORCED SPLIT" FIX ---
            cited_nodes = [n for n in all_nodes if n.eid in original_cited]
            uncited_nodes = [n for n in all_nodes if n.eid not in original_cited]
            
            # Take max 2 cited nodes, fill the rest with uncited distractors (cap at 4 total)
            selected_cited = cited_nodes[:2]
            selected_uncited = uncited_nodes[:(4 - len(selected_cited))]
            top_nodes = selected_cited + selected_uncited
            
            if not top_nodes:
                pbar.update(1)
                return None
                
            node_map = {n.eid: n for n in top_nodes}
            all_eids = list(node_map.keys())
            
            # 1. EXACT SHAPLEY COMPUTATION (2^N combinations)
            subset_results = {}
            all_subsets = []
            for r in range(len(all_eids) + 1):
                all_subsets.extend(itertools.combinations(all_eids, r))
                
            for subset in all_subsets:
                subset_nodes = [node_map[eid] for eid in subset]
                res = await pipeline.generate_answer(question, options, subset_nodes, task_name)
                cop = str(res["gpt_output"].get("cop_index", "-1"))
                subset_results[tuple(sorted(subset))] = 1 if cop == correct_idx else 0

            shapley_scores = calculate_shapley_values(top_nodes, subset_results)

            # 2. POSITIONAL SHIFTING TEST
            golden_eid = max(shapley_scores, key=shapley_scores.get)
            positional_results = {"top": 0, "middle": 0, "bottom": 0, "tested": False}
            
            if shapley_scores[golden_eid] > 0 and len(top_nodes) > 2:
                golden_node = node_map[golden_eid]
                distractors = [n for n in top_nodes if n.eid != golden_eid]
                mid_idx = len(distractors) // 2
                
                positions = {
                    "top": [golden_node] + distractors,
                    "middle": distractors[:mid_idx] + [golden_node] + distractors[mid_idx:],
                    "bottom": distractors + [golden_node]
                }
                
                for pos_name, ordered_nodes in positions.items():
                    res = await pipeline.generate_answer(question, options, ordered_nodes, task_name)
                    cop = str(res["gpt_output"].get("cop_index", "-1"))
                    positional_results[pos_name] = 1 if cop == correct_idx else 0
                    
                positional_results["tested"] = True

            pbar.update(1)
            return {
                "id": q_id,
                "original_cited": list(original_cited),
                "shapley_scores": shapley_scores,
                "positional_results": positional_results
            }

        except Exception as e:
            traceback.print_exc()
            pbar.update(1)
            return None

async def run_shapley(csv_path: str, task_name: str, model_name: str, workers: int, subset_size: int):
    log.info(f"--- RUNNING SHAPLEY & POSITIONAL EVAL FOR: {task_name} ---")
    
    df = pd.read_csv(csv_path)
    safe_model_name = model_name.replace(":", "-")
    eval_file = f"outputs/{task_name}_model_{safe_model_name}.json"
    evidence_file = f"outputs/{task_name}_evidence_{safe_model_name}.json"
    
    if not os.path.exists(eval_file) or not os.path.exists(evidence_file):
        log.error("Missing base output files! Run main.py first.")
        return

    with open(eval_file, 'r') as f: eval_data = {str(item['id']): item for item in json.load(f)}
    with open(evidence_file, 'r') as f: evidence_data = {str(item['id']): item for item in json.load(f)}

    merged_items = []
    for _, row in df.iterrows():
        q_id = str(row['id'])
        if q_id in eval_data and q_id in evidence_data:
            merged_items.append({
                "id": q_id,
                "question": str(row.get('question', '')),
                "options": safe_literal_eval_dict(row.get('options', {})),
                "correct_index": str(row.get('correct_index', '')).strip(),
                "all_evidence": evidence_data[q_id].get("all_evidence", []),
                "evidence_used": evidence_data[q_id].get("evidence_used", [])
            })
            
    merged_items = merged_items[:subset_size]
    
    pipeline = AsyncKGMCQPipeline(None, None, None, None, None, model_name)
    sem = asyncio.Semaphore(workers)
    pbar = tqdm(total=len(merged_items), desc=f"🧠 Shapley Runs", unit="it")
    
    tasks = [process_shapley_row(item, pipeline, sem, pbar, task_name) for item in merged_items]
    results = await asyncio.gather(*tasks)
    pbar.close()
    
    valid_results = [r for r in results if r is not None]
    
    os.makedirs("diagnostics", exist_ok=True)
    with open(f"diagnostics/{task_name}_shapley_{safe_model_name}.json", "w") as f:
        json.dump(valid_results, f, indent=2)

    generate_shapley_report(valid_results)

def generate_shapley_report(results: List[Dict[str, Any]]):
    cited_shapley = []
    uncited_shapley = []
    pos_stats = {"top": 0, "middle": 0, "bottom": 0, "total_tested": 0}
    
    for r in results:
        cited = set(r["original_cited"])
        for eid, score in r["shapley_scores"].items():
            if eid in cited: cited_shapley.append(score)
            else: uncited_shapley.append(score)
            
        pr = r["positional_results"]
        if pr["tested"]:
            pos_stats["total_tested"] += 1
            pos_stats["top"] += pr["top"]
            pos_stats["middle"] += pr["middle"]
            pos_stats["bottom"] += pr["bottom"]

    avg_cited = sum(cited_shapley)/max(len(cited_shapley), 1)
    avg_uncited = sum(uncited_shapley)/max(len(uncited_shapley), 1)
    
    print("\n" + "="*60)
    print(" 🎲 SHAPLEY VALUE & POSITIONAL REPORT ")
    print("="*60)
    print(f"Dataset Size: {len(results)} questions tested\n")
    
    print("1. ATTRIBUTION (SHAPLEY VALUES)")
    print("   Does the model actually rely on what it cites?")
    print(f"   - Average contribution of CITED nodes:   {avg_cited:.4f}")
    print(f"   - Average contribution of UNCITED nodes: {avg_uncited:.4f}")
    
    if avg_cited <= avg_uncited:
        print("   -> 🚨 CRITICAL FINDING: Model is citing evidence that contributes less to the correct answer than ignored evidence! (Post-hoc Hallucination)")
    else:
        print("   -> Finding: Model's citations generally reflect genuine mathematical contribution.")

    if pos_stats["total_tested"] > 0:
        total = pos_stats["total_tested"]
        t_pct = (pos_stats["top"]/total)*100
        m_pct = (pos_stats["middle"]/total)*100
        b_pct = (pos_stats["bottom"]/total)*100
        
        print("\n2. POSITIONAL BIAS ('Lost in the Middle')")
        print("   When the highest-value evidence is shifted, does accuracy drop?")
        print(f"   - Golden Node at TOP:    {t_pct:.1f}% Accuracy")
        print(f"   - Golden Node at MIDDLE: {m_pct:.1f}% Accuracy")
        print(f"   - Golden Node at BOTTOM: {b_pct:.1f}% Accuracy")
        
        diff = max(t_pct, b_pct) - m_pct
        if diff > 10:
            print(f"   -> 🚨 CRITICAL FINDING: Severe 'Lost in the Middle' effect detected ({diff:.1f}% accuracy penalty for middle context).")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--subset_size", type=int, default=50) 
    args = parser.parse_args()
    asyncio.run(run_shapley(args.csv, args.task, args.model, args.workers, args.subset_size))