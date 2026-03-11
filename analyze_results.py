import json
import argparse
import pandas as pd
import numpy as np
import re
from collections import Counter
from pathlib import Path

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_eids_from_gpt_output(gpt_output: dict) -> list:
    """
    Extracts all unique E-IDs (e.g., E1, E12) from the LLM's reasoning text.
    Handles grouped formats like [E2, E7, E13].
    """
    text_blocks = []
    
    # Grab why_correct
    if 'why_correct' in gpt_output:
        text_blocks.append(str(gpt_output['why_correct']))
        
    # Grab all why_others_incorrect explanations
    if 'why_others_incorrect' in gpt_output:
        woi = gpt_output['why_others_incorrect']
        if isinstance(woi, dict):
            text_blocks.extend([str(v) for v in woi.values()])
            
    combined_text = " ".join(text_blocks)
    
    # Regex to find ANY instance of 'E' followed by digits
    matches = re.findall(r"E\d+", combined_text)
    
    # Deduplicate while preserving order
    seen = set()
    unique_eids = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            unique_eids.append(m)
    return unique_eids

def get_dominance_category(routes, ranks):
    if not routes: return "No Evidence"
    evidence_items = list(zip(routes, ranks))
    kg_nodes = [item for item in evidence_items if item[0] == 'kg_sui_concept_def']
    
    if kg_nodes:
        best_kg_rank = min(r for _, r in kg_nodes)
        if best_kg_rank <= 5:
            return "kg_sui_concept_def (Top 5 - Critical)"
        else:
            return "kg_sui_concept_def (Rank >5 - Supporting)"
    
    if any(r == 'kg_semantic_name_only' for r in routes):
        return "kg_semantic_name_only (Name Match)"
        
    return "dense_def_faiss (Vector Search)"

def normalize_evidence_source(evidence_data):
    """Handles both `_evidence_.json` and `_retrieval_cache.json`"""
    ev_map = {}
    if isinstance(evidence_data, list):
        for item in evidence_data:
            ev_map[item['id']] = item.get('all_evidence', [])
    elif isinstance(evidence_data, dict):
        ev_map = evidence_data
    return ev_map

def analyze_pipeline(model_file, evidence_file):
    print(f"\n{'='*60}")
    print(f"📊 ANALYZING: {Path(model_file).name}")
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
        all_nodes = ev_map[q_id]
        
        gpt_out = m_item.get('gpt_output', {})
        predicted = str(gpt_out.get('cop_index', '-1')).strip()
        correct = str(m_item.get('testbed_data', {}).get('correct_index', '')).strip()
        
        is_abstain = (predicted == "-1")
        is_correct = is_abstain if "fake" in model_file else (predicted == correct)

        # --- DYNAMIC EXTRACTION OF EVIDENCE USED ---
        used_eids = extract_eids_from_gpt_output(gpt_out)
        
        all_nodes_sorted = sorted(all_nodes, key=lambda x: x['score'], reverse=True)
        node_lookup = {n['eid']: {'rank': i+1, 'route': n['route'], 'score': n['score']} for i, n in enumerate(all_nodes_sorted)}

        used_ranks = [node_lookup[eid]['rank'] for eid in used_eids if eid in node_lookup]
        used_routes = [node_lookup[eid]['route'] for eid in used_eids if eid in node_lookup]
        used_scores = [node_lookup[eid]['score'] for eid in used_eids if eid in node_lookup]
        
        if is_abstain:
            complexity = "Abstentions"
        elif len(used_ranks) == 1 and used_ranks[0] == 1:
            complexity = "Simple Answers"
        else:
            complexity = "Complex Answers"

        records.append({
            'is_correct': is_correct,
            'is_abstain': is_abstain,
            'complexity': complexity,
            'used_ranks': used_ranks,
            'used_routes': used_routes,
            'avg_used_score': np.mean(used_scores) if used_scores else 0.0,
            'top_avail_score': all_nodes_sorted[0]['score'] if all_nodes_sorted else 0.0,
            'has_kg': any('kg_sui' in r for r in used_routes),
            'evidence_category': get_dominance_category(used_routes, used_ranks)
        })

    df = pd.DataFrame(records)
    total_q = len(df)
    
    if total_q == 0:
        print("No matching data found between model file and evidence/cache file.")
        return

    # 1. Source Distribution
    all_used_routes = [r for sublist in df['used_routes'] for r in sublist]
    print(f"\n1. 🛣️  RAW EVIDENCE SOURCE DISTRIBUTION")
    if all_used_routes:
        counts = Counter(all_used_routes)
        total_cites = sum(counts.values())
        for k, v in counts.most_common():
            print(f"   - {k:<40}: {v:4d} ({v/total_cites*100:5.1f}%)")

    # 2. Accuracy by Source
    print(f"\n2. 🎯 ACCURACY BY EVIDENCE CATEGORY (Strict Rank)")
    categories = ["kg_sui_concept_def (Top 5 - Critical)", "kg_sui_concept_def (Rank >5 - Supporting)", "kg_semantic_name_only (Name Match)", "dense_def_faiss (Vector Search)"]
    print(f"\n   {'CATEGORY':<45} | {'COUNT':<6} | {'ACCURACY':<8}")
    print(f"   {'-'*45} | {'-'*6} | {'-'*8}")
    for cat in categories:
        sub = df[df['evidence_category'] == cat]
        if len(sub) > 0:
            print(f"   {cat:<45} | {len(sub):<6} | {sub['is_correct'].mean()*100:6.2f}%")

    # 3. Complexity Table
    print(f"\n3. 📊 REASONING COMPLEXITY DISTRIBUTION")
    complexity_order = ["Abstentions", "Simple Answers", "Complex Answers"]
    print(f"\n   {'CATEGORY':<20} | {'COUNT':<6} | {'PERCENTAGE':<10}")
    print(f"   {'-'*20} | {'-'*6} | {'-'*10}")
    for cat in complexity_order:
        count = len(df[df['complexity'] == cat])
        print(f"   {cat:<20} | {count:<6} | {count/total_q*100:6.1f}%")

    # 4. Rank Effectiveness
    print(f"\n4. 🏆 EVIDENCE RANKING EFFECTIVENESS")
    complex_df = df[df['complexity'] == "Complex Answers"]
    if not complex_df.empty:
        all_complex_ranks = [r for sublist in complex_df['used_ranks'] for r in sublist]
        num_complex = len(complex_df)
        rank_counts = Counter(all_complex_ranks)
        top_ranks = sorted([(r, (rank_counts.get(r, 0) / num_complex) * 100) for r in range(1, 33)], key=lambda x: x[1], reverse=True)
        print(f"   - Top 3 Evidences             : " + ", ".join([f"E{r} ({v:.1f}%)" for r, v in top_ranks[:3]]))
        q_using_top5 = complex_df['used_ranks'].apply(lambda x: any(r <= 5 for r in x)).sum()
        print(f"   - Usage of top-5 evidences (%) : {q_using_top5 / num_complex * 100:.1f}%")

    # 5. IMPACT ANALYSIS (Fixed for Fake task logic)
    print(f"\n5. 🚀 EVIDENCE IMPACT ON SUCCESS")
    
    # Do not filter out abstentions if this is the fake task, because abstentions ARE the answers
    is_fake_task = "fake" in model_file
    analysis_df = df if is_fake_task else df[~df['is_abstain']]
    
    if not analysis_df.empty:
        # Avoid calculating on empty subsets to prevent NaNs
        kg_subset = analysis_df[analysis_df['has_kg']]
        no_kg_subset = analysis_df[~analysis_df['has_kg']]
        
        kg_acc = kg_subset['is_correct'].mean() * 100 if not kg_subset.empty else 0.0
        no_kg_acc = no_kg_subset['is_correct'].mean() * 100 if not no_kg_subset.empty else 0.0
        
        print(f"   - Knowledge Graph Advantage  : {kg_acc - no_kg_acc:.1f}% Accuracy")
        print(f"     (Acc with KG: {kg_acc:.1f}% vs. Acc without KG: {no_kg_acc:.1f}%)")
        
        # Filter out 0.0 scores (happens if model cited nothing)
        scored_df = analysis_df[analysis_df['avg_used_score'] > 0.0]
        correct_subset = scored_df[scored_df['is_correct']]
        wrong_subset = scored_df[~scored_df['is_correct']]
        
        correct_score = correct_subset['avg_used_score'].mean() if not correct_subset.empty else 0.0
        wrong_score = wrong_subset['avg_used_score'].mean() if not wrong_subset.empty else 0.0
        
        print(f"   - Evidence Confidence Gap    : {correct_score - wrong_score:.3f}")
        print(f"     (Avg Score Correct: {correct_score:.3f} | Avg Score Wrong: {wrong_score:.3f})")
        
        conf_fail = len(wrong_subset[wrong_subset['avg_used_score'] > 0.80])
        print(f"   - High-Confidence Failures   : {conf_fail}")
        print(f"     (Wrong answers despite using evidence score > 0.80)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", required=True, help="Path to Model Output JSON")
    parser.add_argument("--evidence_file", required=True, help="Path to Evidence JSON OR Retrieval Cache JSON")
    args = parser.parse_args()
    analyze_pipeline(args.model_file, args.evidence_file)