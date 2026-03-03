import argparse
import logging
import datetime
import os
import pandas as pd
import random
import sys
import tempfile
import shutil
from _utils import ProfilerBase, configure_logging, add_common_args

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(CURRENT_DIR)

try:
    from core.text_to_speech import tts_factory, MODEL_CHOICES
except ImportError as e:
    print(f"Error importing TTS models: {e}")
    sys.exit(1)


class TTSProfiler(ProfilerBase):
    def __init__(self, model_names, voices=None, n_threads=1):
        logger = logging.getLogger("TTSProfiler")
        super().__init__(model_names, logger, run_forever=False, n_threads=n_threads)
        self.models = {}
        self.voices = voices or {}
        self.temp_dir = tempfile.mkdtemp(prefix="tts_profile_")
        self._load_models(model_names)

    def _load_models(self, model_names):
        """Load TTS models"""
        for model_name in model_names:
            try:
                voice = self.voices.get(model_name, None)
                model = tts_factory(model_name, voice)
                self.models[model_name] = model
                self._logger.info(f"Loaded TTS model: {model_name} (voice: {model.voice})")
            except Exception as e:
                self._logger.error(f"Failed to load model {model_name}: {e}")

    def _get_inference_time(self, model_name: str, input_item: str | None = None) -> float:
        """Run TTS synthesis and return time in seconds"""
        if input_item is None:
            input_item = "これはテストです。"
        
        model = self.models.get(model_name)
        if model is None:
            return 0.0
        
        try:
            output_file = os.path.join(self.temp_dir, f"output_{model_name}.wav")
            
            # Measure synthesis time using perf_counter for accurate wall-clock measurement
            import time
            start_time = time.perf_counter()
            model.synthesize_to_wav(input_item, output_file)
            elapsed_time = time.perf_counter() - start_time
            
            # Clean up output file to save space
            if os.path.exists(output_file):
                os.remove(output_file)
            
            return elapsed_time
        except Exception as e:
            self._logger.warning(f"TTS synthesis error: {e}")
            return 0.0

    def _cleanup(self, model_name: str):
        import gc
        gc.collect()

    def __del__(self):
        """Cleanup temporary directory"""
        
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                self._logger.warning(f"Failed to cleanup temp directory: {e}")


def load_text_samples(csv_path, sample_size=10):
    """Load random text samples from CSV file"""
    if not csv_path or not os.path.exists(csv_path):
        logging.warning(f"Path {csv_path} not found. Using default Japanese text.")
        return [
            "これはテストです。",
            "こんにちは、世界。",
            "テキスト音声合成のプロファイリングを開始します。",
            "日本語のテキストを音声に変換します。",
            "機械学習モデルの性能測定です。"
        ] * max(1, sample_size // 5)
    
    df = pd.read_csv(csv_path)
    # Try to find a text column (Question, Text, Sentence, etc.)
    text_columns = [col for col in df.columns if col.lower() in ['question', 'text', 'sentence', 'content']]
    
    if text_columns:
        sentences = df[text_columns[0]].dropna().tolist()
    else:
        # Use first column as fallback
        sentences = df.iloc[:, 0].dropna().tolist()
    
    return random.sample(sentences, min(len(sentences), sample_size))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profiling for Text-to-Speech Models")
    add_common_args(
        parser,
        model_choices=MODEL_CHOICES,
        default_model="tsuki",
        default_input="これはテストです。",
        input_desc="Japanese text for TTS synthesis"
    )
    
    parser.add_argument(
        "--voice",
        type=str,
        action="append",
        dest="voices",
        help="Voice for each model (format: model_name:voice_name). Can be used multiple times."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="washing_machine_question.csv",
        help="CSV file with text data for TTS profiling"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of samples to use from CSV (default: %(default)s)"
    )
    
    args = parser.parse_args()
    configure_logging(args.logging)

    # Parse voice assignments
    voice_map = {}
    if args.voices:
        for voice_spec in args.voices:
            if ":" in voice_spec:
                model, voice = voice_spec.split(":", 1)
                voice_map[model] = voice
    
    # Load text samples
    text_samples = load_text_samples(args.csv, args.sample_size)
    
    if text_samples:
        print(f"\n>>> Starting profiling for TTS models with {len(text_samples)} text samples...")
        
        output_file = f"profile_tsuki_{datetime.date.today().strftime('%Y%m%d')}.csv"
        
        profiler = TTSProfiler(
            args.models,
            voices=voice_map,
            n_threads=args.threads
        )
        
        profiler.profile_models(
            n_iters=args.repeat,
            inputs=text_samples,
            store=output_file,
            print_stats=True
        )
    else:
        print("No text samples available for profiling.")
