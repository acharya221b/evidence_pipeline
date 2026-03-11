import pandas as pd
import json
import os
import logging
import re
import numpy as np

class FullDataEval:
    def __init__(self, folder_name, all_files, correct_score=1, incorrect_score=-0.25):
        self.folder_name = folder_name
        self.correct_score = correct_score
        self.incorrect_score = incorrect_score
        self.all_files = all_files
        logging.info(f"Evaluator initialized with {len(self.all_files)} files to process.")

    def read_json(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Could not read JSON {file_path}. Error: {e}")
            return []

    def calculate_score(self, correct, wrong):
        return (correct * self.correct_score) + (wrong * self.incorrect_score)
    
    def _evaluate_reasoning_sample(self, sample, task_name):
        """
        Evaluates a sample based on the task type.
        """
        try:
            predicted = str(sample.get('gpt_output', {}).get('cop_index', '')).strip()
            
            # --- LOGIC FOR FAKE TASK ---
            if "reasoning_fake" in task_name:
                # For fake questions, the model MUST abstain.
                # In our system, Abstain/I don't know is explicitly mapped to "-1".
                return predicted == "-1"
            
            # --- LOGIC FOR STANDARD TASKS ---
            else:
                correct = str(sample.get('testbed_data', {}).get('correct_index', '')).strip()
                if not predicted or not correct: 
                    return False
                return predicted == correct
                
        except Exception:
            return False

    def _evaluate_single_file(self, file_path, task_name, model_name, subset_size):
        all_data = self.read_json(file_path)
        if not all_data:
            return None

        total_correct = 0
        total_wrong = 0
        subset_accuracies = []
        
        # SUBSET LOGIC
        for i in range(0, len(all_data), subset_size):
            subset = all_data[i : i + subset_size]
            
            # Pass task_name down to the sample evaluator
            subset_correct = sum(1 for sample in subset if self._evaluate_reasoning_sample(sample, task_name))
            
            if len(subset) > 0:
                accuracy = (subset_correct / len(subset)) * 100
                subset_accuracies.append(accuracy)
            
            total_correct += subset_correct
            
        total_wrong = len(all_data) - total_correct

        return {
            "total_correct": total_correct,
            "total_wrong": total_wrong,
            "subset_accuracies": subset_accuracies
        }

    def run_all_evaluations(self, subset_size=100):
        main_report_data = []
        subset_map = {} 
        
        for file_path in self.all_files:
            filename = os.path.basename(file_path)
            # Regex to parse filenames like: reasoning_fake_model_phi4-14b.json
            pattern = r'(.+?)_model_(.+?)\.json'
            match = re.search(pattern, filename)
            
            if not match:
                logging.warning(f"Could not parse filename '{filename}'. Skipping.")
                continue
            
            task_name, model_name = match.groups()
            kg_rag_status = 'yes'

            eval_results = self._evaluate_single_file(file_path, task_name, model_name, subset_size)
            if eval_results:
                total_correct = eval_results["total_correct"]
                total_wrong = eval_results["total_wrong"]
                total = total_correct + total_wrong
                overall_accuracy = (total_correct / total * 100) if total > 0 else 0
                
                accuracies = eval_results["subset_accuracies"]
                
                # Store subsets with specific key
                subset_key = f"{task_name}_{model_name}"
                subset_map[subset_key] = accuracies

                avg_subset_accuracy = np.mean(accuracies) if accuracies else 0.0
                std_dev_subset_accuracy = np.std(accuracies) if accuracies else 0.0
                
                report_row = {
                    'model_name': model_name,
                    'task_name': task_name,
                    'kg_rag': kg_rag_status,
                    'total_samples': total,
                    'correct': total_correct,
                    'wrong': total_wrong,
                    'overall_accuracy_%': f"{overall_accuracy:.2f}",
                    'avg_subset_accuracy_%': f"{avg_subset_accuracy:.2f}",
                    'std_dev_subset_accuracy': f"{std_dev_subset_accuracy:.2f}",
                    'penalty_score': self.calculate_score(total_correct, total_wrong),
                    'subset_size_used': subset_size
                }
                main_report_data.append(report_row)
        
        return pd.DataFrame(main_report_data), subset_map