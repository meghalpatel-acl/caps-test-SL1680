import argparse
import logging
import datetime
import os
import pandas as pd
import random
import sys
from _utils import ProfilerBase, configure_logging, add_common_args

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(CURRENT_DIR)
try:
    from core.intent_classifier.intent_classification import predict_onnx
except ImportError as e:
    print(f"Error importing classifier: {e}")
    sys.exit(1)

class WashingBERTProfiler(ProfilerBase):
    def __init__(self, model_names, n_threads=1):
        logger = logging.getLogger("WashingBERTProfiler")
        super().__init__(model_names, logger, run_forever=False, n_threads=n_threads)
        

    def _get_inference_time(self, model_name: str, input_item: str | None = None) -> float:
      
        res = predict_onnx(input_item)
        return (res["latency_ms"] / 1000) if res else 0.0

    def _cleanup(self, model_name: str):
        import gc
        gc.collect()

def load_random_csv_data(csv_path, sample_size=10):
    if not csv_path or not os.path.exists(csv_path):
        logging.warning(f"Path {csv_path} not found. Skipping.")
        return []
    df = pd.read_csv(csv_path)
    sentences = df['Question'].dropna().tolist()
    return random.sample(sentences, min(len(sentences), sample_size))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequential Profiling for Bert")
    add_common_args(
    parser,
    model_choices=[],      # or remove entirely if optional
    default_model="Bert",    # SINGLE value, not list
    default_input="None",
    input_desc="None"
)
    # Arguments for separate CSV paths
    parser.add_argument("--csv", type=str, default="washing_machine_question.csv", help="CSV for Bert test input")
    
    parser.add_argument("--sample-size", type=int, default=50)
        
    args = parser.parse_args()
    configure_logging(args.logging)

    model_name = "Bert"  # or args.model if add_common_args exposes it

    csv_path = args.csv
    test_inputs = load_random_csv_data(csv_path, args.sample_size)

    if test_inputs:
        print(f"\n>>> Starting profiling for {model_name} with {len(test_inputs)} samples...")

        output_file = f"profile_{model_name}_{datetime.date.today().strftime('%Y%m%d')}.csv"

        profiler = WashingBERTProfiler(
            [model_name],        
            args.threads
        )

        profiler.profile_models(
            n_iters=1,
            inputs=test_inputs,
            store=output_file,
            print_stats=True
        )
