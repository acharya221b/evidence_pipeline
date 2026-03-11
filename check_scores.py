import json
import argparse
import pandas as pd
import numpy as np
import re
from pathlib import Path

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_eids_from_gpt_output(gpt_output):
    """
    Extracts all unique E-IDs (e.g., E1, E12) from the LLM's reasoning text.
    """
    text_blocks = []
    if isinstance(gpt_output, dict):
        if 'why_correct' in gpt_output:
            text_blocks.append(str(gpt_output['why_correct']))
        if 'why_others_incorrect' in gpt_output:
            woi = gpt_output['why_others_incorrect']
            if isinstance(woi, dict):
                text_blocks.extend([str(v) for v in woi.values()])
    
    combined_text = " ".join(text_blocks)
    matches = re.findall(r"E\d+", combined_text)
    return list(set(matches))

def normalize_evidence_source(evidence_data):
    """
    Handles both `_evidence_.json` (list of dicts) 
    and `_retrieval_cache.json` (dict of lists).
    Returns: { "question_id": [ {eid, score, ...} ] }
    """
    ev_map = {}
    if isinstance(evidence_data, list):
        # Format: outputs/task_evidence_model.json
        for item in evidence_data:
            ev_map[item['id']] = item.get('all_evidence', [])
    elif isinstance(evidence_data, dict):
        # Format: outputs/task_retrieval_cache.json
        ev_map = evidence_data
    return ev_map

def analyze_scores(model_file, evidence_file):
    print(f"\n{'='*60}")
    print(f"📈 SCORE CALIBRATION: {Path(model_file).name}")
    print(f"{'='*60}")

    try:
        model_data = load_json(model_file)
        evidence_data = load_json(evidence_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    ev_map = normalize_evidence_source(evidence_data)
    records = []

    for m_item in model_data:
        q_id = m_item['id']
        if q_id not in ev_map: continue
        
        gpt_out = m_item.get('gpt_output', {})
        predicted = str(gpt_out.get('cop_index', '-1')).strip()
        correct = str(m_item.get('testbed_data', {}).get('correct_index', '')).strip()
        
        # Determine Correctness (Handle Fake task logic if needed)
        if "fake" in model_file:
            is_correct = (predicted == "-1")
        else:
            # Skip abstentions for calibration analysis (we want to know: when it answers, is it right?)
            if predicted == "-1": continue
            is_correct = (predicted == correct)
        
        # --- DYNAMIC EXTRACTION ---
        # 1. Try explicit list first
        used_eids = gpt_out.get('evidence_used', [])
        # 2. If empty/missing, regex parse the text
        if not used_eids:
            used_eids = extract_eids_from_gpt_output(gpt_out)
        
        # Get Max Score of used evidence
        all_nodes = ev_map[q_id]
        # Handle cache format vs evidence file format variations
        # Cache usually has dicts inside the list
        scores = []
        for n in all_nodes:
            # Check if EID is in used list
            if n.get('eid') in used_eids:
                scores.append(float(n.get('score', 0.0)))
        
        # If no evidence cited, we can't calibrate score
        if not scores: continue
            
        max_score = max(scores)
        records.append({'is_correct': is_correct, 'score': max_score})

    if not records:
        print("No answered questions with cited evidence found.")
        return

    df = pd.DataFrame(records)
    
    # Create bins
    bins = [0.0, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ['< 0.60', '0.60 - 0.69', '0.70 - 0.79', '0.80 - 0.89', '> 0.90']
    df['score_bin'] = pd.cut(df['score'], bins=bins, labels=labels)
    
    print(f"\n{'SCORE BRACKET':<15} | {'QUESTIONS':<10} | {'ACCURACY':<10}")
    print("-" * 40)
    
    for label in reversed(labels):
        sub_df = df[df['score_bin'] == label]
        if not sub_df.empty:
            acc = sub_df['is_correct'].mean() * 100
            print(f"{label:<15} | {len(sub_df):<10} | {acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--evidence_file", required=True, help="Evidence JSON or Retrieval Cache JSON")
    args = parser.parse_args()
    analyze_scores(args.model_file, args.evidence_file)