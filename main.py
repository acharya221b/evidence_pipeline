import os
import json
import logging
import argparse
import asyncio
import glob
from pathlib import Path
import math

import pandas as pd
from tqdm import tqdm  # Switched to standard tqdm for manual control
from sentence_transformers import SentenceTransformer, CrossEncoder
import traceback
import config
from utils import safe_literal_eval_dict, FaissIndex
from schemas import EvidenceNode
from clients import ThreadSafeNebulaClient, AsyncOllamaClient
from pipeline import AsyncKGMCQPipeline
from evaluation.evaluator import FullDataEval

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger("main_runner")
logging.getLogger("nebula3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- HELPER: Reconstruct Node ---
def dict_to_node(d):
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

# --- UPDATED: process_row now updates the global progress bar ---
async def process_row(row, pipeline, sem, cache, cache_lock, pbar, task_name=None):
    async with sem:
        q_id = str(row.get('id', 'unknown'))
        question = str(row.get('question', ''))
        options = safe_literal_eval_dict(row.get('options', {}))
        correct_idx = str(row.get('correct_index', '')).strip()
        
        try:
            nodes = []
            is_cached = False
            if q_id in cache:
                raw_list = cache[q_id]
                nodes = [dict_to_node(d) for d in raw_list]
                is_cached = True
            
            if not is_cached:
                nodes = await pipeline.retrieve_evidence(question, options)
                cache_nodes_data = [{
                    "eid": n.eid, "score": n.score01, "route": n.route,
                    "text": n.text, "meta": n.meta, "trace": n.trace
                } for n in nodes]
                
                async with cache_lock:
                    cache[q_id] = cache_nodes_data

            result = await pipeline.generate_answer(question, options, nodes, task_name)
            
            eval_item = {
                "id": q_id,
                "testbed_data": {"correct_index": correct_idx},
                "gpt_output": result["gpt_output"]
            }
            
            evidence_item = {
                "id": q_id,
                "evidence_used": result["evidence_payload"]["evidence_used"],
                "all_evidence": result["evidence_payload"]["all_evidence"]
            }
            
            # Successfully finished one item
            pbar.update(1)
            return eval_item, evidence_item

        except Exception as e:
            log.warning(f"Failed to process QID {q_id}: {e}. Skipping row.")
            # Still update pbar so percentage remains accurate
            traceback.print_exc()
            pbar.update(1)
            return None

async def run_batch(csv_path: str, task_name: str, model_name: str, max_concurrent: int, subset_size: int, eval_only: bool = False):

    # ==========================================
    # EVALUATION ONLY MODE
    # ==========================================
    if eval_only:
        log.info(f"--- RUNNING EVALUATION ONLY FOR TASK: {task_name} ---")
        all_task_files = glob.glob(f"outputs/{task_name}_model_*.json")
        
        if not all_task_files:
            log.error(f"No output files found matching outputs/{task_name}_model_*.json")
            return
            
        log.info(f"Found {len(all_task_files)} model files for evaluation.")
        
        evaluator = FullDataEval(folder_name="rag_results", all_files=all_task_files)
        report_df, subset_map = evaluator.run_all_evaluations(subset_size=subset_size)
        
        report_csv = f"outputs/{task_name}_evaluation_report.csv"
        report_df.to_csv(report_csv, index=False)
        
        subset_json_path = f"outputs/{task_name}_subset_accuracies.json"
        existing_subs = {}
        if os.path.exists(subset_json_path):
            try:
                with open(subset_json_path, 'r') as f: existing_subs = json.load(f)
            except: pass
        existing_subs.update(subset_map)
        with open(subset_json_path, 'w') as f: json.dump(existing_subs, f, indent=4)
            
        log.info(f"Evaluation complete. CSV saved to {report_csv}")
        print("\n--- FINAL COMPARATIVE REPORT ---")
        print(report_df.to_string())
        return # EXIT HERE. Do not run the rest of the pipeline.

    # 1. Load Data
    try:
        df = pd.read_csv(csv_path)
    except:
        df = pd.read_csv(csv_path, sep=None, engine='python')
    
    log.info(f"Loaded {len(df)} total rows from {csv_path}")

    os.makedirs("outputs", exist_ok=True)
    safe_model_name = model_name.replace(":", "-")
    eval_file = f"outputs/{task_name}_model_{safe_model_name}.json"
    evidence_file = f"outputs/{task_name}_evidence_{safe_model_name}.json"
    cache_path = f"outputs/{task_name}_retrieval_cache.json"

    # 2. RESUME LOGIC
    processed_ids = set()
    existing_eval_data = []
    existing_evidence_data = []

    if os.path.exists(eval_file) and os.path.exists(evidence_file):
        try:
            with open(eval_file, 'r') as f:
                existing_eval_data = json.load(f)
                processed_ids = {str(item['id']) for item in existing_eval_data}
            with open(evidence_file, 'r') as f:
                existing_evidence_data = json.load(f)
            log.info(f"Resuming... found {len(processed_ids)} completed.")
        except: pass

    # Filter out already done questions
    df_remaining = df[~df['id'].astype(str).isin(processed_ids)]
    
    if len(df_remaining) == 0:
        log.info("All questions already processed!")
    else:
        log.info(f"Processing remaining {len(df_remaining)} questions...")

        # 3. Load Models
        log.info("Loading Embedding and CrossEncoder models to CUDA...")
        embedder = SentenceTransformer(config.EMBED_MODEL, device="cuda")
        cross = CrossEncoder(config.CROSS_ENCODER_MODEL, device="cuda")
        
        log.info("Loading FAISS indices...")
        sem_faiss = FaissIndex(str(Path(config.DATA_DIR)/config.SEM_INDEX_FILE), str(Path(config.DATA_DIR)/config.SEM_META_JSONL))
        def_faiss = FaissIndex(str(Path(config.DATA_DIR)/config.DEF_INDEX_FILE), str(Path(config.DATA_DIR)/config.DEF_META_FILE))

        nebula = ThreadSafeNebulaClient(config.NEBULA_HOST, config.NEBULA_PORT, config.NEBULA_USER, config.NEBULA_PASSWORD, config.NEBULA_SPACE)
        #llm = AsyncOllamaClient(config.OPENAI_BASE_URL, model_name, config.API_KEY)

        
        pipeline = AsyncKGMCQPipeline(sem_faiss, def_faiss, embedder, cross, nebula, model_name)

        # 4. Load Cache
        retrieval_cache = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f: retrieval_cache = json.load(f)
                log.info(f"Loaded {len(retrieval_cache)} items from cache.")
            except: pass
        
        # 5. INITIALIZE CONTINUOUS PROGRESS BAR
        rows_to_process = df_remaining.to_dict('records')
        total_to_process = len(rows_to_process)
        
        # description shows model name, unit is iterations
        pbar = tqdm(total=total_to_process, desc=f"🚀 {model_name}", unit="it")

        CHUNK_SIZE = 50 
        sem = asyncio.Semaphore(max_concurrent)
        cache_lock = asyncio.Lock()

        # 6. CHUNKED LOOP (Saves every 50, but bar is continuous)
        for i in range(0, total_to_process, CHUNK_SIZE):
            chunk = rows_to_process[i : i + CHUNK_SIZE]
            #current_chunk_num = (i // CHUNK_SIZE) + 1
            
            #log.info(f"--- Processing Chunk {current_chunk_num}/{total_chunks} ({len(chunk)} items) ---")
            # Pass pbar to the tasks
            tasks = [process_row(row, pipeline, sem, retrieval_cache, cache_lock, pbar, task_name) for row in chunk]
            results = await asyncio.gather(*tasks) # silent gather
            
            valid_results = [r for r in results if r is not None]
            
            # Append to in-memory lists
            existing_eval_data.extend([r[0] for r in valid_results])
            existing_evidence_data.extend([r[1] for r in valid_results])
            
            # IMMEDIATE SAVE TO DISK
            #log.info(f"Saving progress... ({len(existing_eval_data)} total)")
            with open(eval_file, "w", encoding="utf-8") as f:
                json.dump(existing_eval_data, f, indent=2)
            with open(evidence_file, "w", encoding="utf-8") as f:
                json.dump(existing_evidence_data, f, indent=2)
            
            # Save Cache periodically
            if not os.path.exists(cache_path) or len(retrieval_cache) > 0:
                with open(cache_path, 'w') as f: json.dump(retrieval_cache, f, indent=2)

        pbar.close()
        log.info(f"Successfully processed {len(existing_eval_data)} total items.")

    # 7. EVALUATION
    log.info(f"Gathering all evaluated models for task '{task_name}'...")
    all_task_files = glob.glob(f"outputs/{task_name}_model_*.json")
    evaluator = FullDataEval(folder_name="rag_results", all_files=all_task_files)
    report_df, subset_map = evaluator.run_all_evaluations(subset_size=subset_size)
    
    report_csv = f"outputs/{task_name}_evaluation_report.csv"
    report_df.to_csv(report_csv, index=False)
    
    subset_json_path = f"outputs/{task_name}_subset_accuracies.json"
    existing_subs = {}
    if os.path.exists(subset_json_path):
        try:
            with open(subset_json_path, 'r') as f: existing_subs = json.load(f)
        except: pass
    existing_subs.update(subset_map)
    with open(subset_json_path, 'w') as f: json.dump(existing_subs, f, indent=4)
        
    log.info(f"Evaluation complete. CSV saved to {report_csv}")
    print("\n--- FINAL COMPARATIVE REPORT ---")
    print(report_df.to_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Async RAG Pipeline Runner")
    parser.add_argument("--csv", required=False, default="", help="Path to input CSV file") # Changed to required=False
    parser.add_argument("--task", required=True, help="Task name")
    parser.add_argument("--model", required=False, default="none", help="LLM Model name") # Changed to required=False
    parser.add_argument("--workers", type=int, default=5, help="Concurrency")
    parser.add_argument("--subset_size", type=int, default=100, help="Subset size")
    parser.add_argument("--eval_only", action="store_true", help="Skip retrieval/generation and only run evaluation")
    
    args = parser.parse_args()
    asyncio.run(run_batch(args.csv, args.task, args.model, args.workers, args.subset_size, args.eval_only))