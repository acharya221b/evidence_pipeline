import subprocess
import sys
import time

# ================= CONFIGURATION =================
# List the models you want to evaluate sequentially
MODELS_TO_RUN = [
    # "gemma2:27b",
    # "phi4:14b",
    "llama3.3:70b",
    "deepseek-r1:14b",
    "mistral:7b-instruct",
    "llama3.2:latest"
    
]

# Shared parameters for the task
CSV_PATH = "/home/macharya/dev/medkg-eval/data/reasoning_nota.csv"
TASK_NAME = "reasoning_nota"
WORKERS = 5
SUBSET_SIZE = 100 #37
# =================================================

def run_pipeline_for_model(model_name):
    print(f"\n{'='*60}")
    print(f"🚀 STARTING PIPELINE FOR MODEL: {model_name}")
    print(f"{'='*60}\n")

    command = [
        sys.executable, "main.py",
        "--csv", CSV_PATH,
        "--task", TASK_NAME,
        "--model", model_name,
        "--workers", str(WORKERS),
        "--subset_size", str(SUBSET_SIZE)
    ]

    try:
        # Check=True will raise an error if the script fails, stopping the chain
        # Set Check=False if you want to continue to the next model even if one fails
        subprocess.run(command, check=True)
        print(f"\n✅ FINISHED: {model_name}\n")
        
        # Optional: Sleep briefly to ensure OS reclaims resources completely
        time.sleep(5)
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: Pipeline failed for {model_name}. Exit code: {e.returncode}")
        # Uncomment the next line if you want to stop the whole experiment on error
        # sys.exit(1) 

def main():
    total_start = time.time()
    
    for i, model in enumerate(MODELS_TO_RUN, 1):
        print(f"Processing {i}/{len(MODELS_TO_RUN)}...")
        run_pipeline_for_model(model)

    total_end = time.time()
    duration = (total_end - total_start) / 60
    
    print(f"{'='*60}")
    print(f"🎉 ALL MODELS COMPLETED in {duration:.2f} minutes.")
    print(f"Check outputs/{TASK_NAME}_evaluation_report.csv for the combined results.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()