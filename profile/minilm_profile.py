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
    from core.embeddings.minilm import MiniLMSynap, MiniLMLlama
except ImportError as e:
    print(f"Error importing MiniLM models: {e}")
    sys.exit(1)


class MiniLMProfiler(ProfilerBase):
    def __init__(self, model_names, model_type="llama", n_threads=1):
        logger = logging.getLogger("MiniLMProfiler")
        super().__init__(model_names, logger, run_forever=False, n_threads=n_threads)
        self.model_type = model_type
        self.models = {}
        self._n_threads = n_threads
        self._load_models(model_names)

    def _load_models(self, model_names):
        """Load MiniLM models based on model_names"""
        for model_name in model_names:
            if self.model_type == "synap":
                self.models[model_name] = MiniLMSynap(
                    model_name=model_name,
                    hf_model=model_name,
                    model_path="models/gguf/granite-embedding-107m-multilingual-GGUF/model.gguf",  
                    normalize=False
                )
            elif self.model_type == "llama":
                self.models[model_name] = MiniLMLlama(
                    model_name=model_name,
                    model_path=f"{PROJECT_ROOT}/models/gguf/granite-embedding-107m-multilingual-Q8_0.gguf",
                    normalize=False,
                    n_threads=self._n_threads
                )

    def _get_inference_time(self, model_name: str, input_item: str | None = None) -> float:
        """Run inference and return time in seconds"""
        if input_item is None:
            input_item = "これはテストです"
        
        model = self.models.get(model_name)
        if model is None:
            return 0.0
        # print(input_item)
        _ = model.generate(input_item)
        infer_time = model.last_infer_time
        return infer_time if infer_time else 0.0

    def _cleanup(self, model_name: str):
        import gc
        gc.collect()


def load_random_csv_data(csv_path, sample_size=10):
    """Load random samples from CSV file"""
    if not csv_path or not os.path.exists(csv_path):
        logging.warning(f"Path {csv_path} not found. Using default text.")
        return ["これはテストです"] * sample_size
    
    df = pd.read_csv(csv_path)
    sentences = df['Question'].dropna().tolist() if 'Question' in df.columns else df.iloc[:, 0].dropna().tolist()
    return random.sample(sentences, min(len(sentences), sample_size))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profiling for MiniLM Embeddings Models")
    add_common_args(
        parser,
        model_choices=["minilm-llama", "minilm-synap"],
        default_model="minilm-llama",
        default_input="これはテストです",
        input_desc="Japanese text for embedding generation"
    )
    
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["llama", "synap"],
        default="llama",
        help="Type of MiniLM model to use (default: %(default)s)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="washing_machine_question.csv",
        help="CSV file for test inputs"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of samples to use from CSV (default: %(default)s)"
    )
    
    args = parser.parse_args()
    configure_logging(args.logging)

    # Load test inputs
    test_inputs = load_random_csv_data(args.csv, args.sample_size)
    
    if test_inputs:
        print(f"\n>>> Starting profiling for MiniLM with {len(test_inputs)} samples...")
        
        output_file = f"profile_minilm_{args.model_type}_{datetime.date.today().strftime('%Y%m%d')}.csv"
        
        profiler = MiniLMProfiler(
            args.models,
            model_type=args.model_type,
            n_threads=args.threads
        )
        
        profiler.profile_models(
            n_iters=args.repeat,
            inputs=test_inputs,
            store=output_file,
            print_stats=True
        )
