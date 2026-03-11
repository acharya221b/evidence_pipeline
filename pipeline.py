import os
import asyncio
import json
from platform import system
import re
import time
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from json_repair import repair_json
from streamlit import user
from schemas import EvidenceNode, MCQZipAnswer
from dotenv import load_dotenv
#from openai import OpenAI
from openai import AsyncOpenAI
from utils import (
    build_joint_query, build_question_only_query, sigmoid, minmax01,
    mmr_select, bm25_scores, prune_nodes, normalize_text, 
    sanitize_zip_answer, extract_first_json_obj, 
    extract_eids_from_text, safe_sort_key
)

class AsyncKGMCQPipeline:
    def __init__(self, sem_faiss, def_faiss, embedder, cross, nebula, model_name):
        self.sem_faiss = sem_faiss
        self.def_faiss = def_faiss
        self.embedder = embedder
        self.cross = cross
        self.nebula = nebula
        self.model_name = model_name

        load_dotenv()
        self.llm = AsyncOpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("API_KEY"))
        
        # EXACT PARAMS FROM GOLD STANDARD
        self.semantic_topk_dense = 500 #220
        self.seed_k_mmr = 140 #60
        self.dense_def_topk = 160 #160 #80
        self.dense_defs_keep = 40 #40 #20
        self.kg_defs_keep = 30 #40 #50 #25
        self.semantic_name_keep = 20 #16 #20 #10
        self.prune_keep_k = 32
        self.kg_min_sigmoid_to_enforce = 0.70
        self.retry_strict_once = True
        self.require_kg_def_if_available = True

    def _blocking_retrieval(self, question: str, options: Dict[str, str]) -> List[EvidenceNode]:
        joint_query = build_joint_query(question, options) if options else build_question_only_query(question)
        
        # # 1. Retrieve Seeds
        q_emb = self.embedder.encode([joint_query], normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        D_sem, I_sem = self.sem_faiss.search(q_emb, self.semantic_topk_dense)
        
        valid_sem_indices = []
        valid_sem_scores = []
        for d, i in zip(D_sem, I_sem):
            if int(i) >= 0:
                valid_sem_indices.append(int(i))
                valid_sem_scores.append(float(d))
        
        cand_meta = [self.sem_faiss.meta[i] for i in valid_sem_indices]
        cand_name = [m.get("name") or m.get("NAME") or "" for m in cand_meta]
        cand_sui = [m.get("sui") or m.get("SUI") or "" for m in cand_meta]
        
        name_embs = self.embedder.encode([normalize_text(n) for n in cand_name], normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        mmr_idx = mmr_select(q_emb[0], name_embs, np.array(valid_sem_scores, dtype=np.float32), k=self.seed_k_mmr)
        seeds_mmr = [{"sui": str(cand_sui[idx]), "name": str(cand_name[idx])} for idx in mmr_idx]
        
        # 2. Semantic Nodes
        top_dense = [{"sui": str(cand_sui[i]), "name": str(cand_name[i])} for i in range(min(self.semantic_name_keep * 6, len(cand_sui)))]
        sem_texts = [f"Semantic: {normalize_text(it['name'])}\n(SUI={it['sui']})" for it in top_dense]
        sem_bm = bm25_scores(joint_query, sem_texts)
        sem_bm01 = minmax01(sem_bm)
        
        sem_nodes = []
        sorted_sem = sorted(zip(top_dense, sem_bm01.tolist()), key=lambda x: x[1], reverse=True)[:self.semantic_name_keep * 3]
        for it, s01 in sorted_sem:
            sem_nodes.append(EvidenceNode(
                eid="", route="kg_semantic_name_only", score01=float(s01),
                text=f"Semantic: {normalize_text(it['name'])}\n(SUI={it['sui']})",
                meta={"sui": it["sui"], "name": it["name"]},
                trace={"route": "kg_semantic_name_only", "seed_sui": it["sui"]}
            ))

        # 3. Dense Defs
        D_def, I_def = self.def_faiss.search(q_emb, self.dense_def_topk)
        dense_defs = []
        for dist, idx in zip(D_def, I_def):
            if idx < 0: continue
            m = self.def_faiss.meta[int(idx)]
            dense_defs.append({
                "DEF": m.get("DEF") or m.get("def_text") or "",
                "ATUI": m.get("ATUI") or "",
                "SAB": m.get("SAB") or ""
            })
            
        dense_nodes = []
        if dense_defs:
            texts = [d["DEF"] for d in dense_defs]
            ce_scores = sigmoid(self.cross.predict([(joint_query, t) for t in texts], show_progress_bar=False))
            bm_scores = minmax01(bm25_scores(joint_query, texts))
            final_scores = 0.75 * ce_scores + 0.25 * bm_scores
            
            ranked_dense = sorted(zip(dense_defs, final_scores), key=lambda x: x[1], reverse=True)[:self.dense_defs_keep]
            for d, s in ranked_dense:
                dense_nodes.append(EvidenceNode(
                    eid="", route="dense_def_faiss", score01=float(s),
                    text=f"Definition: {normalize_text(d['DEF'])}\n(ATUI={d['ATUI']} SAB={d['SAB']})",
                    meta={"ATUI": d["ATUI"], "SAB": d["SAB"]},
                    trace={"route": "dense_def_faiss", "ATUI": d["ATUI"], "SAB": d["SAB"]}
                ))

        # 4. KG Expansion
        seed_suis = [s["sui"] for s in seeds_mmr if s["sui"]]
        lookup = self.nebula.lookup_semantic_vids(seed_suis)
        sem_vids = list(lookup.values())
        sem_to_concepts = self.nebula.sty_reverse_concepts(sem_vids)
        all_concepts = list(set([c for cs in sem_to_concepts.values() for c in cs]))
        pairs = self.nebula.def_pairs_for_concepts(all_concepts)
        def_vids = list(set([p[1] for p in pairs]))
        def_props = self.nebula.fetch_def_props(def_vids)
        
        semvid_to_seed = {v: k for k, v in lookup.items()}
        concept_to_semvid = {}
        for sv, cs in sem_to_concepts.items():
            for c in cs: concept_to_semvid[c] = sv
            
        def_trace = {}
        for cvid, dvid in pairs:
            if dvid not in def_trace:
                sv = concept_to_semvid.get(cvid)
                def_trace[dvid] = {
                    "seed_sui": semvid_to_seed.get(sv), "sem_vid": sv,
                    "concept_vid": cvid, "def_vid": dvid
                }

        kg_nodes = []
        if def_props:
            kg_items = [(vid, p) for vid, p in def_props.items() if p.get("DEF")]
            if kg_items:
                texts = [p["DEF"] for _, p in kg_items]
                ce_scores = sigmoid(self.cross.predict([(joint_query, t) for t in texts], show_progress_bar=False))
                bm_scores = minmax01(bm25_scores(joint_query, texts))
                final_scores = 0.75 * ce_scores + 0.25 * bm_scores
                
                ranked_kg = sorted(zip(kg_items, final_scores), key=lambda x: x[1], reverse=True)[:self.kg_defs_keep]
                
                for (vid, props), s in ranked_kg:
                    dt = def_trace.get(vid, {})
                    kg_nodes.append(EvidenceNode(
                        eid="", route="kg_sui_concept_def", score01=float(s),
                        text=f'Definition: "{normalize_text(props["DEF"])}"\n(ATUI="{props["ATUI"]}" SAB="{props["SAB"]}")',
                        meta={"def_vid": vid, "ATUI": props["ATUI"], "SAB": props["SAB"]},
                        trace={"route": "kg_sui_concept_def", **dt, "atui": props["ATUI"], "sab": props["SAB"]}
                    ))

        # 5. Merge & Prune
        all_nodes = kg_nodes + dense_nodes + sem_nodes
        pruned_nodes = prune_nodes(mode="mcq", question=question, options=options, nodes=all_nodes, keep_k=self.prune_keep_k)
        
        for i, n in enumerate(pruned_nodes, 1):
            n.eid = f"E{i}"
            
        return pruned_nodes

    async def _call_llm_json(self, prompt: str) -> Dict[str, Any]:
        """Internal helper for LLM JSON completions."""
        response = await self.llm.chat.completions.create(
            model=self.model_name, 
            messages=[{"role": "user", "content": prompt}], 
            temperature=0.0, 
            response_format={"type": "json_object"}
        )
        raw_text = response.choices[0].message.content
        return json.loads(repair_json(raw_text))
    
    async def retrieve_evidence(self, question: str, options: Dict[str, str]) -> List[EvidenceNode]:
        return await asyncio.to_thread(self._blocking_retrieval, question, options)

    # --- UPDATED: No Try-Except here (Let it crash so Main can catch) ---
    async def _llm_chat(self, system: str, user: str) -> Dict[str, Any]:
        # NOTE: We DO NOT catch exceptions here anymore. 
        # If LLM fails (504), we want the error to bubble up so we can skip the row.
        raw = await self.llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
        return extract_first_json_obj(raw) or {}

    async def generate_answer(self, question: str, options: Dict[str, str], nodes: List[EvidenceNode], task_name: Optional[str] = None) -> Dict[str, Any]:
        option_keys = sorted(options.keys(), key=safe_sort_key)
        allowed_eids = [n.eid for n in nodes]

        kg_nodes = [n for n in nodes if n.route == "kg_sui_concept_def"]
        kg_def_eids = [n.eid for n in kg_nodes]
        kg_max = max([n.score01 for n in kg_nodes], default=0.0)
        
        enforce_kg = bool(kg_def_eids) and (kg_max >= self.kg_min_sigmoid_to_enforce) and self.require_kg_def_if_available

        evidence_block = "\n\n".join([f"[{n.eid}] {n.text}" for n in nodes])
        options_str = "\\n".join([f"{k}: {v}" for k, v in options.items()])

        system_policy = ""
        if "reasoning_fct" in task_name.lower():
            few_shot_examples = """
            EXAMPLE INPUT:
            Evidence list: [E1] Atenolol is a beta-adrenergic antagonist used for hypertension. [E2] Diuretics increase urine output.
            Question: Which class of medication is Atenolol?
            Options: {"0": "Diuretic", "1": "Beta-blocker", "2": "ACE Inhibitor"}
            EXAMPLE OUTPUT:
            {
            "why_correct": "[E1] explicitly defines Atenolol as a beta-adrenergic antagonist, which is synonymous with beta-blocker.",
            "why_others_incorrect": {
                "0": "Diuretics function by increasing urine production, which is a different mechanism [E2].",
                "2": "ACE Inhibitors are not mentioned in the evidence for Atenolol."
            },
            "cop_index": "1",
            "answer": "Beta-blocker",
            "evidence_used": ["E1", "E2"]
            }
            """
            system_base = ("You are a medical reasoning expert system. Answer based on the 'Evidence list' provided. Give a detailed explanation of why the chosen option is correct and why the other options are not correct. Cite all the top relevant evidence IDs (e.g., [E1][E2]) used for your reasoning statements. The reasoning should match the evidence closely.")

            output_format = ("Your output must be JSON matching this exact key order: {'why_correct': 'step-by-step reasoning with [E#] citations', 'why_others_incorrect': {'0': 'reason with [E#]', ...}, 'cop_index': 'correct index', 'answer': 'option text', 'evidence_used': ['E1', ...]}")

        elif "reasoning_nota" in task_name.lower():
            few_shot_examples = """
            EXAMPLE INPUT:
            Evidence list: [E5] Influenza is a viral infection. [E6] Antibiotics kill bacteria but are ineffective against viruses.
            Question: Which antibiotic is effective against the Influenza virus?
            Options: {"0": "Amoxicillin", "1": "Azithromycin", "2": "None of the above"}
            EXAMPLE OUTPUT:
            {
            "why_correct": "Influenza is a viral infection [E5]. Evidence [E6] states antibiotics do not treat viruses. Because options 0 and 1 are antibiotics, they are ineffective. Therefore, 'None of the above' is the only valid choice.",
            "why_others_incorrect": {
                "0": "Amoxicillin is an antibiotic, which does not treat viruses [E6].",
                "1": "Azithromycin is an antibiotic, ineffective against Influenza [E5]."
            },
            "cop_index": "2",
            "answer": "None of the above",
            "evidence_used": ["E5", "E6"]
            }
            """
            system_base = (
                "You are a strict medical reasoning assistant. \n"
                "RULE 1: Assume all specific options are INCORRECT unless the Evidence explicitly supports them.\n"
                "RULE 2: If the evidence refutes the specific options, or if there is no evidence to support them, you MUST select the 'None of the above' option.\n"
            )
            output_format = ("Your output must be JSON matching this exact key order: {'why_correct': 'reasoning with [E#] citations', 'why_others_incorrect': {'0': 'reason with [E#]'}, 'cop_index': 'index', 'answer': 'text', 'evidence_used': ['E#']}")

        elif "reasoning_fake" in task_name.lower():    
            few_shot_examples = """
            EXAMPLE INPUT:
            Evidence list: [E1] The Humerus is the long bone in the arm. [E2] The Brachial artery supplies the arm.
            Question: Which artery supplies blood to the 'Crystal-Bone' in the human arm?
            Options: {"0": "Brachial Artery", "1": "Radial Artery"} 
            EXAMPLE OUTPUT:
            {
            "why_correct": "There is no anatomical structure known as the 'Crystal-Bone' mentioned in the evidence. The premise of the question is medically invalid and fabricated.",
            "why_others_incorrect": {
                "0": "While the Brachial artery exists [E2], it does not supply a fabricated bone.",
                "1": "The premise is fabricated."
            },
            "cop_index": "-1",
            "answer": "",
            "evidence_used": []
            }
            """
            system_base = (
                "You are an expert medical evaluator tasked with identifying TRICK questions.\n"
                "WARNING: The user will try to trick you with fabricated diseases, fake anatomy, and nonsensical scenarios.\n"
                "RULE 1: Read the question carefully. If the premise of the question is medically invalid, impossible, or completely unsupported by the evidence, you MUST output cop_index='-1'.\n"
                "RULE 2: Do NOT attempt to guess an answer if the question contains fake terminology.\n"
            )
            output_format = ("Your output must be JSON matching this exact key order: {'why_correct': 'explanation of why the question is invalid or supported', 'why_others_incorrect': {'0': 'reason'}, 'cop_index': '-1 or index', 'answer': 'text', 'evidence_used': []}")


        if enforce_kg:
            system_policy = (
                "High-confidence KG definitions are available.\n"
                "You MUST cite at least one kg_sui_concept_def evidence ID (e.g., [E1], [E2]) in your reasoning.\n"
                "If KG definitions do not support a single option, output cop_index=\"-1\" (or 'None of the above' if valid).\n"
            )
            fallback_reason = "Insufficient KG definition evidence to select a single option."
            fallback_ev = (kg_def_eids[:1] if kg_def_eids else [nodes[0].eid] if nodes else None)
        else:
            system_policy=""
            # system_policy = (
            #     "KG definitions are absent or not confident.\n"
            #     "Still answer using the evidence list.\n"
            #     "You MAY use general medical knowledge, but MUST cite at least one evidence item.\n"
            #     "CRITICAL: If the question is fake or evidence is missing, use '-1' (Abstain) or 'None of the above' if available.\n"
            # )
            fallback_reason = "Insufficient evidence to select a single option."
            fallback_ev = ([nodes[0].eid] if nodes else None)

        user_msg = f"""
        {system_base}\n
        {system_policy}\n
        {few_shot_examples}\n\n
        --- CURRENT TASK ---\n
        Evidence list:\n
        {evidence_block}
        Question: {question}\nOptions:\n{options_str}\n\n
        Provide your answer. {output_format}
        """.strip()

        try:
            obj = await self._call_llm_json(user_msg)
        except Exception as e:
            logging.error(f"LLM Call 1 failed: {e}")
            obj = {}

        ans = sanitize_zip_answer(obj, options, allowed_eids, fallback_reason, fallback_ev)

        # --- KG RETRY LOGIC ---
        if enforce_kg and not set(ans.evidence_used).intersection(set(kg_def_eids)):
            if self.retry_strict_once:
                strict_sys = f"FINAL WARNING: Cite at least one KG evidence ID: {kg_def_eids}, OR abstain/pick NOTA."
                try:
                    obj2 = await self._call_llm_json(user_msg + "\n\n" + strict_sys)
                    ans2 = sanitize_zip_answer(obj2, options, allowed_eids, fallback_reason, fallback_ev)
                    if ans2.cop_index == "-1" or set(ans2.evidence_used).intersection(set(kg_def_eids)):
                        ans = ans2
                except: pass

        gpt_out = ans.model_dump(exclude={'evidence'})
        
        all_evidence = []
        for n in nodes:
            item = {
                "eid": n.eid, 
                "score": n.score01, 
                "route": n.route, 
                "text": n.text, 
                "trace": n.trace,
                "meta": n.meta
            }
            if n.meta.get("ATUI"): item["ATUI"] = n.meta["ATUI"]
            if n.meta.get("SAB"): item["SAB"] = n.meta["SAB"]
            all_evidence.append(item)

        evidence_payload = {
            "evidence_used": ans.evidence_used,
            "all_evidence": all_evidence
        }

        return {
            "gpt_output": gpt_out,
            "evidence_payload": evidence_payload
        }