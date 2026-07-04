import os
import json
import logging
import argparse
import asyncio
import pandas as pd
from tqdm import tqdm
import traceback
from typing import List, Dict, Any

from utils import safe_literal_eval_dict
from schemas import EvidenceNode
from pipeline import AsyncKGMCQPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger("explainability")

# --- HELPER: Reconstruct Node ---
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

def deduplicate_nodes(nodes: List[EvidenceNode]) -> List[EvidenceNode]:
    """Simple redundancy removal based on exact/normalized text matches."""
    seen = set()
    unique = []
    for n in nodes:
        clean_text = n.text.strip().lower()
        if clean_text not in seen:
            unique.append(n)
            seen.add(clean_text)
    return unique

async def process_explain_row(item: Dict[str, Any], pipeline: AsyncKGMCQPipeline, sem: asyncio.Semaphore, pbar: tqdm, task_name: str) -> Dict[str, Any]:
    """Runs the 3 interventions for a single question."""
    async with sem:
        try:
            q_id = item['id']
            question = item['question']
            options = item['options']
            correct_idx = str(item['correct_index']).strip()
            
            all_nodes = [dict_to_node(n) for n in item['all_evidence']]
            used_eids = set(item['evidence_used'])
            
            # --- CREATE INTERVENTION CONTEXTS ---
            # (A) Necessity: Remove used evidence
            necessity_nodes = [n for n in all_nodes if n.eid not in used_eids]
            
            # (B) Sufficiency: Keep ONLY used evidence
            sufficiency_nodes = [n for n in all_nodes if n.eid in used_eids]
            
            # (C) Redundancy: Remove semantic duplicates
            redundancy_nodes = deduplicate_nodes(all_nodes)
            
            # --- RE-RUN MODEL FOR EACH VARIANT ---
            results = {}
            for variant_name, nodes in [
                ("necessity", necessity_nodes),
                ("sufficiency", sufficiency_nodes),
                ("redundancy", redundancy_nodes)
            ]:
                res = await pipeline.generate_answer(
                    question=question,
                    options=options,
                    nodes=nodes,
                    task_name=task_name
                )
                
                gpt_out = res["gpt_output"]
                results[variant_name] = {
                    "cop_index": gpt_out.get("cop_index", "-1"),
                    "answer": gpt_out.get("answer", ""),
                    "evidence_used": gpt_out.get("evidence_used", []),
                    "why_correct": gpt_out.get("why_correct", "")
                }
            
            # --- COMPILE LOG RECORD ---
            original_cop = str(item['original_prediction'].get('cop_index', '-1'))
            original_is_correct = (original_cop == correct_idx)
            
            log_record = {
                "id": q_id,
                "question": question,
                "correct_index": correct_idx,
                "original_prediction": original_cop,
                "original_is_correct": original_is_correct,
                "original_evidence_used": list(used_eids),
                "interventions": {}
            }
            
            # Add analytical flags for easy pandas ingestion later
            for variant, out in results.items():
                variant_cop = str(out["cop_index"])
                log_record["interventions"][variant] = {
                    **out,
                    "is_correct": (variant_cop == correct_idx),
                    "answer_changed": (variant_cop != original_cop)
                }

            pbar.update(1)
            return log_record

        except Exception as e:
            log.warning(f"Failed to process explainability for QID {item.get('id')}: {e}")
            traceback.print_exc()
            pbar.update(1)
            return None

async def run_explainability(csv_path: str, task_name: str, model_name: str, workers: int, subset_size: int):
    log.info(f"--- RUNNING EXPLAINABILITY DIAGNOSTICS FOR: {task_name} ---")
    
    # 1. Load Original Dataset
    try:
        df = pd.read_csv(csv_path)
    except:
        df = pd.read_csv(csv_path, sep=None, engine='python')
    
    # 2. Load Original Pipeline Results (Model outputs & Evidence payloads)
    safe_model_name = model_name.replace(":", "-")
    eval_file = f"outputs/{task_name}_model_{safe_model_name}.json"
    evidence_file = f"outputs/{task_name}_evidence_{safe_model_name}.json"
    
    if not os.path.exists(eval_file) or not os.path.exists(evidence_file):
        log.error("Missing original output files. Run main.py first to generate baseline predictions.")
        return

    with open(eval_file, 'r') as f:
        eval_data = {str(item['id']): item for item in json.load(f)}
    with open(evidence_file, 'r') as f:
        evidence_data = {str(item['id']): item for item in json.load(f)}

    # 3. Merge data for the subset
    merged_items = []
    for _, row in df.iterrows():
        q_id = str(row['id'])
        if q_id in eval_data and q_id in evidence_data:
            merged_items.append({
                "id": q_id,
                "question": str(row.get('question', '')),
                "options": safe_literal_eval_dict(row.get('options', {})),
                "correct_index": str(row.get('correct_index', '')).strip(),
                "original_prediction": eval_data[q_id].get("gpt_output", {}),
                "all_evidence": evidence_data[q_id].get("all_evidence", []),
                "evidence_used": evidence_data[q_id].get("evidence_used", [])
            })
            
    if not merged_items:
        log.error("No matching IDs found between CSV and cached outputs.")
        return

    # Subsetting (Take first `subset_size` items)
    merged_items = merged_items[:subset_size]
    log.info(f"Loaded {len(merged_items)} items for Explainability Interventions.")

    # 4. Initialize Lightweight Pipeline
    # Pass None to heavy components (Embedder, CrossEncoder, Faiss, Nebula) since we don't retrieve here.
    pipeline = AsyncKGMCQPipeline(None, None, None, None, None, model_name)
    
    # 5. Run Concurrently
    sem = asyncio.Semaphore(workers)
    pbar = tqdm(total=len(merged_items), desc=f"🔍 Diagnostic Runs", unit="it")
    
    tasks = [process_explain_row(item, pipeline, sem, pbar, task_name) for item in merged_items]
    results = await asyncio.gather(*tasks)
    pbar.close()
    
    valid_results = [r for r in results if r is not None]
    
    # 6. Save Raw Results
    os.makedirs("diagnostics", exist_ok=True)
    out_file = f"diagnostics/{task_name}_explainability_{safe_model_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(valid_results, f, indent=2)
    log.info(f"Raw diagnostics saved to {out_file}")

    # 7. Print Final Aggregated Report
    generate_report(valid_results)

def generate_report(results: List[Dict[str, Any]]):
    """Generates the separated report for Original Correct vs Incorrect."""
    correct_samples = [r for r in results if r['original_is_correct']]
    incorrect_samples = [r for r in results if not r['original_is_correct']]
    
    def calc_stats(samples):
        if not samples: return {"N": 0}
        total = len(samples)
        stats = {"N": total}
        for variant in ["necessity", "sufficiency", "redundancy"]:
            changed = sum(1 for s in samples if s['interventions'][variant]['answer_changed'])
            is_corr = sum(1 for s in samples if s['interventions'][variant]['is_correct'])
            stats[variant] = {
                "changed_pct": round((changed / total) * 100, 1),
                "correct_pct": round((is_corr / total) * 100, 1)
            }
        return stats

    corr_stats = calc_stats(correct_samples)
    inc_stats = calc_stats(incorrect_samples)

    print("\n" + "="*60)
    print(" 📊 EXPLAINABILITY & DIAGNOSTIC REPORT ")
    print("="*60)
    
    print(f"\n✅ ORIGINALLY CORRECT PREDICTIONS (N = {corr_stats['N']})")
    if corr_stats['N'] > 0:
        print(f"  (A) Necessity (Removed Used Evidence):")
        print(f"      - Answer Changed: {corr_stats['necessity']['changed_pct']}% (Ideal: High - indicates model actually relied on the evidence)")
        print(f"  (B) Sufficiency (Kept ONLY Used Evidence):")
        print(f"      - Remained Correct: {corr_stats['sufficiency']['correct_pct']}% (Ideal: High - indicates extracted evidence was sufficient)")
        print(f"  (C) Redundancy (Removed Duplicates):")
        print(f"      - Answer Changed: {corr_stats['redundancy']['changed_pct']}% (Ideal: Low - indicates robustness to context density)")

    print(f"\n❌ ORIGINALLY INCORRECT PREDICTIONS (N = {inc_stats['N']})")
    if inc_stats['N'] > 0:
        print(f"  (A) Necessity (Removed Used Evidence):")
        print(f"      - Answer Changed: {inc_stats['necessity']['changed_pct']}% (If high: Evidence was actively misleading the model)")
        print(f"  (B) Sufficiency (Kept ONLY Used Evidence):")
        print(f"      - Became Correct: {inc_stats['sufficiency']['correct_pct']}% (If high: Irrelevant noise in original context broke the reasoning)")
        print(f"  (C) Redundancy (Removed Duplicates):")
        print(f"      - Became Correct: {inc_stats['redundancy']['correct_pct']}% (If high: Duplicate conflicting context confused the model)")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explainability & Interventions Runner")
    parser.add_argument("--csv", required=True, help="Path to original input CSV file")
    parser.add_argument("--task", required=True, help="Task name (e.g. reasoning_nota)")
    parser.add_argument("--model", required=True, help="LLM Model name")
    parser.add_argument("--workers", type=int, default=5, help="Concurrency for API calls")
    parser.add_argument("--subset_size", type=int, default=200, help="Number of questions to evaluate")
    
    args = parser.parse_args()
    asyncio.run(run_explainability(args.csv, args.task, args.model, args.workers, args.subset_size))