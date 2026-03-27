import streamlit as st
import pandas as pd
import json
import glob
import os
import re
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# --- CUSTOM COLOR PALETTE ---
# Highly distinct colors designed for white backgrounds
# Deep Blue, Bright Orange, Emerald Green, Purple, Crimson, Cyan
DISTINCT_COLORS = [
    "#1f77b4", # Muted Blue
    "#ff7f0e", # Safety Orange
    "#2ca02c", # Cooked Asparagus Green
    "#9467bd", # Muted Purple
    "#d62728", # Brick Red
    "#17becf", # Blue-Teal
    "#e377c2", # Raspberry Yogurt Pink
    "#8c564b"  # Chestnut Brown
]

ROUTE_COLORS = {
    "Dense Retrieval (Vector)": "#1f77b4",        # Blue
    "KG Traversal (Top 5 - Critical)": "#2ca02c", # Green
    "KG Traversal (Rank >5 - Supporting)": "#98df8a", # Light Green
    "KG Entity Match (Name)": "#d62728",          # Red
    "No Evidence": "#7f7f7f"                      # Gray
}

COMPLEXITY_COLORS = {
    "Abstentions": "#ff7f0e",     # Orange
    "Simple Answers": "#17becf",  # Teal
    "Complex Answers": "#9467bd"  # Purple
}

# ==========================================
# 1. HELPER LOGIC
# ==========================================

def extract_eids_from_gpt_output(gpt_output):
    """
    Extracts all unique E-IDs (e.g., E1, E12) from the LLM's reasoning text.
    Handles grouped formats like [E2, E7, E13].
    """
    text_blocks = []
    if isinstance(gpt_output, dict):
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
    """
    Classifies the 'Quality' of evidence based on hierarchy AND rank.
    Hierarchy: kg_sui_concept_def > kg_semantic_name_only > dense_def_faiss
    """
    if not routes: return "No Evidence"
    evidence_items = list(zip(routes, ranks))
    
    kg_nodes = [item for item in evidence_items if item[0] == 'kg_sui_concept_def']
    
    if kg_nodes:
        best_kg_rank = min(r for _, r in kg_nodes)
        if best_kg_rank <= 5:
            return "KG Traversal (Top 5 - Critical)"
        else:
            return "KG Traversal (Rank >5 - Supporting)"
    
    if any(r == 'kg_semantic_name_only' for r in routes):
        return "KG Entity Match (Name)"
        
    return "Dense Retrieval (Vector)"

def normalize_evidence_source(evidence_data):
    """
    Handles both `_evidence_.json` (list of dicts) 
    and `_retrieval_cache.json` (dict of lists).
    Returns a unified dict: { "question_id": [list of nodes] }
    """
    ev_map = {}
    if isinstance(evidence_data, list):
        for item in evidence_data:
            ev_map[item['id']] = item.get('all_evidence', [])
    elif isinstance(evidence_data, dict):
        ev_map = evidence_data
    return ev_map

def determine_complexity(predicted, used_ranks):
    if predicted == "-1":
        return "Abstentions"
    elif len(used_ranks) == 1:
        return "Simple Answers"
    else:
        return "Complex Answers"

@st.cache_data
def load_and_process_data():
    """
    Loads ALL JSON files in outputs/ and processes them.
    """
    files = glob.glob("outputs/*_model_*.json")
    all_records = []
    
    # Pre-load caches to avoid re-reading them loop
    cache_files = glob.glob("outputs/*_retrieval_cache.json")
    caches = {}
    for c_file in cache_files:
        try:
            t_name = os.path.basename(c_file).replace("_retrieval_cache.json", "")
            with open(c_file, 'r') as f:
                caches[t_name] = json.load(f)
        except: pass

    for model_file in files:
        filename = os.path.basename(model_file)
        match = re.match(r"(.+)_model_(.+)\.json", filename)
        if not match: continue
        
        task_name = match.group(1)
        model_name = match.group(2)
        
        # --- EVIDENCE LOADING STRATEGY ---
        raw_evidence_data = None
        
        # 1. Try Direct Evidence File
        ev_file = model_file.replace("_model_", "_evidence_")
        if os.path.exists(ev_file):
            try:
                with open(ev_file, 'r') as f: raw_evidence_data = json.load(f)
            except: pass
        
        # 2. Try Cache (Exact Match)
        if not raw_evidence_data and task_name in caches:
            raw_evidence_data = caches[task_name]
            
        # 3. Try Cache (Fuzzy Match)
        if not raw_evidence_data:
            for c_name, c_data in caches.items():
                if c_name in task_name:
                    raw_evidence_data = c_data
                    break
        
        if not raw_evidence_data: continue

        # Normalize the evidence data using the helper
        ev_map = normalize_evidence_source(raw_evidence_data)

        # Load Model Predictions
        try:
            with open(model_file, 'r') as f: m_data = json.load(f)
        except: continue

        for m_item in m_data:
            q_id = m_item['id']
            if q_id not in ev_map: continue
            
            all_nodes = ev_map[q_id]
            gpt_out = m_item.get('gpt_output', {})
            
            predicted = str(gpt_out.get('cop_index', '-1')).strip()
            correct = str(m_item.get('testbed_data', {}).get('correct_index', '')).strip()
            
            is_abstain = (predicted == "-1")
            
            if "fake" in task_name:
                is_correct = is_abstain
            else:
                is_correct = (predicted == correct)

            # --- EXTRACT EVIDENCE USED FROM TEXT ---
            used_eids = extract_eids_from_gpt_output(gpt_out)
            
            all_nodes_sorted = sorted(all_nodes, key=lambda x: x['score'], reverse=True)
            node_lookup = {n['eid']: {'rank': i+1, 'route': n['route'], 'score': n['score']} 
                           for i, n in enumerate(all_nodes_sorted)}

            used_routes = []
            used_ranks = []
            used_scores = []
            
            for eid in used_eids:
                if eid in node_lookup:
                    used_routes.append(node_lookup[eid]['route'])
                    used_ranks.append(node_lookup[eid]['rank'])
                    used_scores.append(node_lookup[eid]['score'])
            
            has_kg = any('kg_sui' in r for r in used_routes)
            avg_score = np.mean(used_scores) if used_scores else 0.0
            max_score = max(used_scores) if used_scores else 0.0
            
            cat = get_dominance_category(used_routes, used_ranks)
            complexity = determine_complexity(predicted, used_ranks)
            
            # Capture Score Profile (Top 32 Scores for Decay Chart)
            score_profile = [n['score'] for n in all_nodes_sorted[:32]]
            while len(score_profile) < 32: score_profile.append(0.0)
            
            all_records.append({
                "Task": task_name,
                "Model": model_name,
                "ID": q_id,
                "Is Correct": is_correct,
                "Is Abstain": is_abstain,
                "Has Evidence Data": bool(all_nodes),
                "Has KG Trace": has_kg,
                "Avg Evidence Score": avg_score,
                "Max Used Score": max_score,
                "Used Scores": used_scores,
                "Score Profile": score_profile,
                "Evidence Category": cat,
                "Complexity": complexity,
                "Used Ranks": used_ranks,
                "Used Routes": used_routes
            })
            
    return pd.DataFrame(all_records)

# ==========================================
# 2. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Medical KG-RAG Analytics", layout="wide")
st.title("🧬 Medical KG-RAG Deep Analytics")

if st.sidebar.button("🔄 Refresh Data"):
    load_and_process_data.clear()

df = load_and_process_data()

if df.empty:
    st.error("No valid data loaded. Check 'outputs/' folder.")
    st.stop()

# Filters
tasks = sorted(df['Task'].unique())
selected_task = st.sidebar.selectbox("Select Task", tasks)
task_df = df[df['Task'] == selected_task]

models = sorted(task_df['Model'].unique())
selected_models = st.sidebar.multiselect("Models", models, default=models)

if not selected_models:
    st.stop()

filtered_df = task_df[task_df['Model'].isin(selected_models)]

# Dynamically assign colors to the selected models to keep them consistent across tabs
MODEL_COLORS = {mod: DISTINCT_COLORS[i % len(DISTINCT_COLORS)] for i, mod in enumerate(selected_models)}

# ==========================================
# 3. TABS
# ==========================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Accuracy", 
    "📊 Reasoning", 
    "🏆 Top Evidences Used", 
    "🚀 Impact & Calibration"
])

# --- TAB 1: ACCURACY ---
with tab1:
    st.subheader("Accuracy by Evidence Category")
    st.caption("Shows how often the model answers correctly based on the highest-quality evidence route it relied upon. Look for 'KG Traversal' outperforming 'Dense Retrieval'.")
    # Display Total Questions Analyzed
    total_unique_questions = len(filtered_df['ID'].unique())
    st.markdown(f"**Total Questions Analyzed:** `{total_unique_questions}`")
    
    # Group and aggregate (mean for accuracy, count for volume)
    cat_stats = filtered_df.groupby(['Model', 'Evidence Category'])['Is Correct'].agg(
        Accuracy='mean',
        Count='count'
    ).reset_index()
    
    cat_stats['Accuracy (%)'] = cat_stats['Accuracy'] * 100
    
    order = ["KG Traversal (Top 5 - Critical)", "KG Traversal (Rank >5 - Supporting)", "KG Entity Match (Name)", "Dense Retrieval (Vector)", "No Evidence"]
    
    fig = px.bar(
        cat_stats, x="Evidence Category", y="Accuracy (%)", color="Model", barmode="group",
        text_auto='.1f', category_orders={"Evidence Category": order},
        title=f"Accuracy vs. Evidence Category - {selected_task}",
        color_discrete_map=MODEL_COLORS, # Use custom model colors
        hover_data={
            "Evidence Category": False,  # Hide from hover since it's on the X-axis
            "Count": True,               # Show the total count of questions for this bar
            "Accuracy (%)": ':.2f'
        }
    )
    st.plotly_chart(fig, use_container_width=True)
    

    st.divider()
    st.subheader("Model Stability (Subset Variance)")
    st.caption("Shows the distribution of accuracy across small subsets of the data. Smaller boxes and tighter clusters mean the model is highly consistent.")
    
    subset_file = f"outputs/{selected_task}_subset_accuracies.json"
    if os.path.exists(subset_file):
        with open(subset_file, 'r') as f:
            subset_data = json.load(f)
            
        box_records = []
        for model in selected_models:
            key = f"{selected_task}_{model}"
            if key in subset_data:
                for acc in subset_data[key]:
                    box_records.append({"Model": model, "Accuracy (%)": acc})
                    
        if box_records:
            box_df = pd.DataFrame(box_records)
            fig_box = px.box(
                box_df, x="Model", y="Accuracy (%)", color="Model",
                points="all", # Shows the individual subset dots
                color_discrete_map=MODEL_COLORS
            )
            st.plotly_chart(fig_box, use_container_width=True)
            

# --- TAB 2: REASONING & EVIDENCE RECIPE ---
with tab2:
    # We create two main columns for the side-by-side layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Reasoning Complexity")
        st.caption("'Simple' means it only needed one evidence. 'Complex' means it synthesized multiple evidence snippets. 'Abstentions' means it rejected the question.")
        comp_counts = filtered_df.groupby(['Model', 'Complexity']).size().reset_index(name='Count')
        comp_counts['Percentage'] = comp_counts.groupby('Model')['Count'].transform(lambda x: x / x.sum() * 100)
        fig_comp = px.bar(
            comp_counts, x="Model", y="Percentage", color="Complexity", 
            text_auto='.1f', category_orders={"Complexity": ["Abstentions", "Simple Answers", "Complex Answers"]},
            color_discrete_map=COMPLEXITY_COLORS,
            text='Percentage',
            title=f"Answer Complexity Distribution - {selected_task}"
        )
        fig_comp.update_traces(texttemplate='%{text:.2f}', textposition='outside', cliponaxis=False)
        st.plotly_chart(fig_comp, use_container_width=True)

    with col2:
        st.subheader("Average Evidence Count: Correct vs. Incorrect")
        st.caption("Stacked bar shows citation count by route. Incorrect answers are faded for easy comparison.")

        all_routes_raw = ['dense_def_faiss', 'kg_semantic_name_only', 'kg_sui_concept_def']
        plot_data = []
        
        # Calculate Averages for both Correct and Incorrect
        for outcome in ['Correct', 'Incorrect']:
            df_to_analyze = filtered_df[filtered_df['Is Correct']] if outcome == 'Correct' else filtered_df[~filtered_df['Is Correct']]
            
            for model in selected_models:
                model_df = df_to_analyze[df_to_analyze['Model'] == model]
                num_q = len(model_df)
                if num_q == 0: continue

                route_counts = Counter([r for sublist in model_df['Used Routes'] for r in sublist])
                for route in all_routes_raw:
                    avg_count = route_counts.get(route, 0) / num_q
                    if avg_count > 0:
                        plot_data.append({
                            'Model': model,
                            'Outcome': outcome,
                            'Route': route,
                            'Avg. Evidence Count': avg_count
                        })

        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            
            route_map_display = {
                'dense_def_faiss': 'Dense Retrieval (Vector)',
                'kg_sui_concept_def': 'KG Traversal',
                'kg_semantic_name_only': 'KG Entity Match (Name)'
            }
            plot_df['Route_Display'] = plot_df['Route'].map(route_map_display)

            # Create Faceted Stacked Bar
            fig_avg = px.bar(
                plot_df, 
                x='Model', 
                y='Avg. Evidence Count', 
                color='Route_Display',
                facet_col='Outcome', 
                barmode='stack', 
                text='Avg. Evidence Count',
                title=f"{selected_task}- Avg Citations per Question (Correct vs Incorrect)",
                color_discrete_map=ROUTE_COLORS
            )
            
            # --- FIX: Apply Opacity to Incorrect Bars safely ---
            # Plotly creates traces where the 'name' is the Route, but we need to know the facet (Outcome).
            # The easiest way to apply formatting per-facet is using for_each_trace
            for trace in fig_avg.data:
                # Plotly stores the facet information in the xaxis/yaxis mapping internally,
                # but 'legendgroup' often contains the color variable, and 'xaxis' contains the subplot.
                # A safer way to handle opacity in facet plots is to map it via an explicit column, 
                # but since we want the legend to stay clean, we use a manual data match.
                
                # Because Plotly groups traces by color (Route), we must update the marker color array directly
                # based on the underlying data points for that trace.
                # Note: px.bar handles opacity natively if we just tell it to!
                pass # We will handle opacity via update_traces below for simplicity

            # Apply Opacity using Plotly's powerful selector
            # 'col=2' targets the second facet column (which is 'Incorrect' if 'Correct' is col 1)
            # This is much safer than parsing trace names.
            fig_avg.update_traces(marker=dict(opacity=0.7), selector=dict(xaxis='x2'))

            # Formatting (Bold & Outside labels)
            fig_avg.update_traces(
                texttemplate='%{text:.2f}', 
                textposition='outside', 
                cliponaxis=False
            )
            
            fig_avg.update_layout(
                yaxis_title="Total Avg Citations Used",
                font=dict(size=14, color="black"),
                title_font=dict(size=20, weight='bold'),
                margin=dict(t=80, b=50, l=50, r=50)
            )
            
            # Bold Annotations (The titles above each facet: "Outcome=Correct")
            fig_avg.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font=dict(size=16, weight='bold')))
            
            # Bold Axis Labels
            fig_avg.update_xaxes(tickfont=dict(weight='bold', size=12), title_font=dict(weight='bold', size=14))
            fig_avg.update_yaxes(tickfont=dict(weight='bold', size=12), title_font=dict(weight='bold', size=14))
            
            st.plotly_chart(fig_avg, use_container_width=True)
        else:
            st.info("Insufficient data to compare Correct vs. Incorrect evidence recipes.")

    # with col2:
    #     st.subheader("Average Evidence Distribution by Route")
    #     st.caption("Shows the average number of evidence used for each route. Switch between filters to see if models change their research strategy when they get an answer wrong.")
        
    #     outcome_filter = st.radio("Filter by Answer Outcome:", ('All', 'Correct', 'Incorrect'), horizontal=True, key='outcome_filter')
        
    #     df_to_analyze = filtered_df.copy()
    #     if outcome_filter == 'Correct': df_to_analyze = filtered_df[filtered_df['Is Correct']]
    #     elif outcome_filter == 'Incorrect': df_to_analyze = filtered_df[~filtered_df['Is Correct']]

    #     all_routes_raw = ['dense_def_faiss', 'kg_semantic_name_only', 'kg_sui_concept_def']
    #     plot_data = []
        
    #     for model in selected_models:
    #         model_df = df_to_analyze[df_to_analyze['Model'] == model]
    #         num_questions = len(model_df)
    #         if num_questions == 0: continue

    #         all_routes_for_model = [route for sublist in model_df['Used Routes'] for route in sublist]
    #         route_counts = Counter(all_routes_for_model)

    #         for route in all_routes_raw:
    #             total_route_citations = route_counts.get(route, 0)
    #             avg_count = total_route_citations / num_questions
    #             if avg_count > 0:
    #                 plot_data.append({'Model': model, 'Route': route, 'Avg. Evidence Count': avg_count})

    #     if plot_data:
    #         plot_df = pd.DataFrame(plot_data)
    #         route_map_display = {
    #             'dense_def_faiss': 'Dense Retrieval (Vector)',
    #             'kg_sui_concept_def': 'KG Traversal',
    #             'kg_semantic_name_only': 'KG Entity Match (Name)'
    #         }
    #         plot_df['Route'] = plot_df['Route'].map(route_map_display)

    #         fig_avg = px.bar(
    #             plot_df, x='Model', y='Avg. Evidence Count', color='Route',
    #             barmode='stack', text='Avg. Evidence Count',
    #             title=f"Avg Citations by Route- {selected_task} - {outcome_filter} Answers",
    #             color_discrete_map=ROUTE_COLORS
    #         )
    #         fig_avg.update_traces(texttemplate='%{text:.2f}', textposition='outside', cliponaxis=False)
    #         fig_avg.update_layout(yaxis_title="Total Avg Citations Used", margin=dict(t=50, b=50, l=50, r=50), uniformtext_minsize=8, uniformtext_mode='hide')
    #         st.plotly_chart(fig_avg, use_container_width=True)
        
    #     else:
    #         st.info(f"No '{outcome_filter}' answers to display for this selection.")

# --- TAB 3: TOP EVIDENCES ---
with tab3:
    st.subheader("Top Evidences Used")
    st.caption("For complex questions, how often did the model include at least one piece of evidence from the Top 5 most relevant results?")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        complex_df = filtered_df[filtered_df['Complexity'] == "Complex Answers"]
        if not complex_df.empty:
            for model in selected_models:
                m_comp = complex_df[complex_df['Model'] == model]
                if len(m_comp) > 0:
                    using_top5 = m_comp['Used Ranks'].apply(lambda x: any(r <= 5 for r in x)).sum()
                    perc = (using_top5 / len(m_comp)) * 100
                    st.metric(label=f"{model} Top-5 Usage", value=f"{perc:.1f}%")
            
        else:
            st.info("No complex questions.")

    with col2:
        st.markdown("##### 📉 Average Score per Evidence")
        st.caption("Tracks the average retriever score from Rank 1 to 32. A steep drop-off indicates the retriever has high confidence in its top results.")
        decay_data = []
        for model in selected_models:
            m_df = filtered_df[filtered_df['Model'] == model]
            if not m_df.empty:
                matrix = np.vstack(m_df['Score Profile'].values)
                means = np.mean(matrix, axis=0)
                for r, score in enumerate(means):
                    decay_data.append({"Model": model, "Rank": r+1, "Avg Score": score})
        
        if decay_data:
            df_decay = pd.DataFrame(decay_data)
            fig_decay = px.line(
                df_decay, x="Rank", y="Avg Score", color="Model", 
                markers=True, title=f"Retrieval Confidence Decay- {selected_task}",
                color_discrete_map=MODEL_COLORS # Use custom model colors
            )
            st.plotly_chart(fig_decay, use_container_width=True)
            

    if not complex_df.empty:
        rank_data = []
        for _, row in complex_df.iterrows():
            for r in row['Used Ranks']:
                rank_data.append({"Model": row['Model'], "Rank": r})
        r_df = pd.DataFrame(rank_data)
        fig_hist = px.histogram(
            r_df, x="Rank", color="Model", barmode="overlay", nbins=32, 
            title=f"Evidence Rank Distribution (Complex)- {selected_task}",
            color_discrete_map=MODEL_COLORS
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption("Shows which ranks the model actually cited during complex queries. A strong left-skew means high-ranked evidence is being utilized.")
        

# --- TAB 4: IMPACT & CALIBRATION ---
with tab4:
    st.subheader("Evidence Impact on Success")
    
    # 1. Aggregate Data for Graphs
    impact_records = []
    for model in selected_models:
        m_df = filtered_df[filtered_df['Model'] == model]
        is_fake = "fake" in selected_task
        analysis_df = m_df if is_fake else m_df[~m_df['Is Abstain']]
        
        # A. KG Advantage Calc
        kg_acc, no_kg_acc = 0.0, 0.0
        kg_count, no_kg_count = 0, 0
        
        if not analysis_df.empty:
            kg_sub = analysis_df[analysis_df['Has KG Trace']]
            no_kg_sub = analysis_df[~analysis_df['Has KG Trace']]
            if not kg_sub.empty: kg_acc = kg_sub['Is Correct'].mean() * 100
            if not no_kg_sub.empty: no_kg_acc = no_kg_sub['Is Correct'].mean() * 100
        adv = kg_acc - no_kg_acc
            
        # B. High Conf Success Calc (UPDATED LOGIC)
        success_rate = 0.0
        num_success = 0
        num_total_high = 0
        
        scored = analysis_df[analysis_df['Max Used Score'] > 0]
        if not scored.empty:
            def is_high_confidence(row):
                scores = row['Used Scores']
                if not scores: return False
                high_scores = sum(1 for s in scores if s > 0.80)
                return (high_scores / len(scores)) >=0.60

            high_conf_total = scored[scored.apply(is_high_confidence, axis=1)]
            num_total_high = len(high_conf_total)
            if num_total_high > 0:
                # Count SUCCESSES instead of failures
                num_success = len(high_conf_total[high_conf_total['Is Correct']])
                success_rate = (num_success / num_total_high) * 100
                
        impact_records.append({
            "Model": model,
            "KG Advantage (%)": adv,
            "KG Used (Count)": kg_count,
            "No KG Used (Count)": no_kg_count,
            "KG Acc (%)": kg_acc,
            "No KG Acc (%)": no_kg_acc,
            "High Conf Success Rate (%)": success_rate, # Updated Key
            "High Conf Successes (Count)": num_success, # Updated Key
            "High Conf Total": num_total_high
        })

    impact_df = pd.DataFrame(impact_records)
    
    # 2. Display Graphs Side-by-Side
    if not impact_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            fig_kg = px.bar(
                impact_df, x="Model", y="KG Advantage (%)", color="Model",
                title=f"Knowledge Graph Advantage (+/- Accuracy %) - {selected_task}",
                text_auto=".1f",
                hover_data={
                    "Model": False,
                    "KG Advantage (%)": ':.1f',
                    "KG Acc (%)": ':.1f',
                    "No KG Acc (%)": ':.1f',
                    "KG Used (Count)": True,
                    "No KG Used (Count)": True
                },
                color_discrete_map=MODEL_COLORS # Use custom model colors
            )
            # Add a zero line to easily see positive vs negative impact
            fig_kg.add_hline(y=0, line_dash="dash", line_color="#333333", line_width=2)
            
            st.caption("Difference in accuracy when using KG Traversal vs. relying on Dense Retrieval.")
            st.plotly_chart(fig_kg, use_container_width=True)

        with col2:
            # Updated to show Success Rate instead of Failure Rate
            fig_success = px.bar(
                impact_df, x="Model", y="High Conf Success Rate (%)", color="Model",
                title=f"High-Confidence Success Rate - {selected_task}",
                text_auto=".1f",
                hover_data={"High Conf Successes (Count)": True, "High Conf Total": True},
                color_discrete_map=MODEL_COLORS # Use custom model colors
            )
            
            st.caption("Percentage of questions where the model answered correctly when atleast 60% of evidence scores > 0.80.")
            st.plotly_chart(fig_success, use_container_width=True)

    st.divider()
    
    # # --- CALIBRATION CHART (Unchanged logic, updated colors) ---
    # st.subheader("📈 Evidence Score Calibration")
    # calib_data = []
    # bins = [0.0, 0.6, 0.7, 0.8, 0.9, 1.0]
    # labels = ['< 0.60', '0.60 - 0.69', '0.70 - 0.79', '0.80 - 0.89', '> 0.90']
    
    # for model in selected_models:
    #     m_df = filtered_df[filtered_df['Model'] == model].copy()
    #     if "fake" not in selected_task: m_df = m_df[~m_df['Is Abstain']]
        
    #     m_df['Score Bin'] = pd.cut(m_df['Max Used Score'], bins=bins, labels=labels)
    #     bin_stats = m_df.groupby('Score Bin', observed=False).agg(
    #         Accuracy=('Is Correct', 'mean'), 
    #         Count=('Is Correct', 'count')
    #     ).reset_index()
        
    #     bin_stats['Accuracy'] = bin_stats['Accuracy'] * 100
    #     bin_stats['Model'] = model
    #     calib_data.append(bin_stats)
        
    # if calib_data:
    #     calib_df = pd.concat(calib_data)
    #     fig = go.Figure()
        
    #     for model in selected_models:
    #         model_data = calib_df[calib_df['Model'] == model]
    #         # Custom transparent bars for count
    #         fig.add_trace(go.Bar(
    #             x=model_data['Score Bin'], y=model_data['Count'], 
    #             name=f"{model} (Count)", opacity=0.4, yaxis='y2',
    #             marker_color=MODEL_COLORS.get(model)
    #         ))
            
    #     for model in selected_models:
    #         model_data = calib_df[calib_df['Model'] == model]
    #         # Custom solid lines for accuracy
    #         fig.add_trace(go.Scatter(
    #             x=model_data['Score Bin'], y=model_data['Accuracy'], 
    #             name=f"{model} (Accuracy %)", mode='lines+markers', line=dict(width=3),
    #             marker_color=MODEL_COLORS.get(model)
    #         ))
            
    #     fig.update_layout(
    #         title=f"Accuracy vs. Max Evidence Score Bracket - {selected_task}",
    #         xaxis_title="Max Evidence Score Bracket",
    #         yaxis=dict(
    #             title=dict(text="Accuracy (%)", font=dict(color="#333333")), 
    #             range=[0, 105], 
    #             tickfont=dict(color="#333333")
    #         ),
    #         yaxis2=dict(
    #             title=dict(text="Number of Questions", font=dict(color="gray")), 
    #             overlaying='y', 
    #             side='right', 
    #             showgrid=False, 
    #             tickfont=dict(color="gray")
    #         ),
    #         legend=dict(x=1.1, y=1, xanchor="left", yanchor="top"),
    #         hovermode="x unified",
    #         margin=dict(r=150)
    #     )
        
    #     st.caption("Compares the number of questions per score bracket against the model's accuracy (lines). An upward slope indicates a well-calibrated system where higher scores lead to better answers.")
    #     st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.subheader("📈 Evidence Score Calibration (Scatter Plot)")
    col_chart, col_text = st.columns([1, 1]) 
    st.markdown("Plots the average accuracy for evidence scores grouped to the nearest 0.01.")
    
    calib_data = []
    
    for model in selected_models:
        m_df = filtered_df[filtered_df['Model'] == model].copy()
        if "fake" not in selected_task: 
            m_df = m_df[~m_df['Is Abstain']]
        
        # 1. Round to 2 decimal places to create a cloud of distinct averages
        m_df['Rounded Score'] = m_df['Max Used Score'].round(2)
        
        # 2. Group by that rounded score and calculate Average Accuracy
        score_stats = m_df.groupby('Rounded Score', observed=False).agg(
            Accuracy=('Is Correct', 'mean'), 
            Count=('Is Correct', 'count')
        ).reset_index()
        
        score_stats['Accuracy (%)'] = score_stats['Accuracy'] * 100
        score_stats['Model'] = model
        calib_data.append(score_stats)
        
    if calib_data:
        calib_df = pd.concat(calib_data)
        
        # --- Create Scatter Plot ---
        fig_calib = px.scatter(
            calib_df, 
            x="Rounded Score", 
            y="Accuracy (%)", 
            size="Count",          # Bigger dot = More questions got this exact score
            color="Model",
            opacity=0.6,           # Increased transparency so overlapping clouds are visible
            hover_name="Model",
            hover_data={
                "Model": False,
                "Rounded Score": ':.2f',
                "Accuracy (%)": ':.1f',
                "Count": True
            },
            title=f"Accuracy vs. Evidence Score (Aggregated by 0.01) - {selected_task}",
            color_discrete_map=MODEL_COLORS,
            size_max=20           # Limit max size so the scatter effect isn't swallowed by giant bubbles
        )

                # Apply Bold Formatting & Resize for 50% width
        fig_calib.update_layout(
            autosize=True,
            width=600,  # Fixed width for the left half
            height=400,
            xaxis_title="Evidence Score (Rounded to 0.01)",
            yaxis_title="Average Accuracy (%)",
            yaxis=dict(range=[-5, 105]),
            xaxis=dict(range=[0.35, 1.05], tickmode='linear', dtick=0.1),
            font=dict(size=12, color="black", family="Arial"),
            title_font=dict(size=16, weight='bold'),
            margin=dict(t=50, b=50, l=10, r=10)
        )
        
        with col_chart:
            st.plotly_chart(fig_calib, use_container_width=False) # False ensures it respects width=600
        
        with col_text:
            st.markdown("### Interpretation")
            st.write("""
            The scatter plot shows how the retriever's confidence score correlates with actual answer accuracy:
            - **X-Axis:** The evidence score (rounded to 0.01).
            - **Y-Axis:** Average accuracy for all questions with that specific score.
            - **Bubble Size:** Represents the number of questions in that score bucket.
            
            **What to look for:**
            - **Positive Correlation:** As the bubble moves right (higher confidence), the accuracy should trend upward.
            - **Density:** If many bubbles are concentrated at the higher end, your retriever is successfully finding high-quality evidence.
            """)
        
        # # Apply Bold Formatting
        # fig_calib.update_layout(
        #     xaxis_title="Evidence Score (Rounded to 0.01)",
        #     yaxis_title="Average Accuracy (%)",
        #     yaxis=dict(range=[-5, 105]),
        #     xaxis=dict(
        #         range=[0.35, 1.05], 
        #         tickmode='linear',
        #         dtick=0.1
        #     ),
        #     font=dict(size=14, color="black", family="Arial"),
        #     title_font=dict(size=20, weight='bold'),
        #     xaxis_title_font=dict(size=16, weight='bold'),
        #     xaxis_tickfont=dict(size=14, weight='bold'),
        #     yaxis_title_font=dict(size=16, weight='bold'),
        #     yaxis_tickfont=dict(size=14, weight='bold'),
        #     legend=dict(title_font=dict(weight='bold'), font=dict(weight='bold')),
        #     margin=dict(t=80, b=50)
        # )
        
        # st.plotly_chart(fig_calib, use_container_width=True)
        # st.caption("🔍 **How to read:** Every dot represents a specific evidence score (e.g., 0.84). The Y-axis is the average accuracy of all questions that received that exact score. The size of the dot shows the volume of questions. An upward-sloping 'cloud' indicates strong calibration.")