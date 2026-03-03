import argparse
import logging
import datetime
import os
import sys
import numpy as np
import soundfile as sf
from typing import Final
from _utils import ProfilerBase, configure_logging, add_common_args

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(CURRENT_DIR)

try:
    from core.speech_to_text.moonshine import MoonshineOnnx, MoonshineSynap, MODEL_CHOICES
except ImportError as e:
    print(f"Error importing Moonshine models: {e}")
    sys.exit(1)

MODEL_TYPES: Final = [
    "onnx",
    "synap"
]


class MoonshineProfiler(ProfilerBase):
    def __init__(self, model_names, n_threads=1):
        logger = logging.getLogger("MoonshineProfiler")
        super().__init__(model_names, logger, run_forever=False, n_threads=n_threads)
        self.models = {}
        self.audio_cache = {}  # Cache for wav files
        self._load_models(model_names)

    def _load_models(self, model_names):
        """Load Moonshine models"""
        for model_name in model_names:
            try:
                model_type, model_size, model_quant = model_name.split("-")
                if model_type == "onnx":
                    self.models[model_name] = MoonshineOnnx(
                        model_size=model_size,
                        quant_type=model_quant,
                        local_model_dir="models/UsefulSensores"
                    )
                elif model_type == "synap":
                    self.models[model_name] = MoonshineSynap(
                        model_size=model_size,
                        quant_type=model_quant,
                        local_model_dir="models/UsefulSensores"
                    )
            except Exception as e:
                self._logger.error(f"Failed to load model {model_name}: {e}")

    def _get_inference_time(self, model_name: str, input_item: str | None = None) -> float:
        """Run inference and return time in seconds"""
        model = self.models.get(model_name)
        if model is None:
            return 0.0
        
        # Use audio from cache or generate silence
        if isinstance(input_item, str) and input_item in self.audio_cache:
            audio = self.audio_cache[input_item]
        else:
            # Use default 5 second silence (16kHz * 5s)
            audio = np.zeros(5 * 16000, dtype="float32")
        
        try:
            _ = model.transcribe(audio)
            infer_time = model.last_infer_time
            return infer_time if infer_time else 0.0
        except Exception as e:
            self._logger.warning(f"Inference error: {e}")
            return 0.0

    def _cleanup(self, model_name: str):
        import gc
        gc.collect()

    def cache_audio_files(self, wav_files):
        """Load and cache audio files in memory"""
        for wav_file in wav_files:
            if os.path.exists(wav_file):
                try:
                    audio, sr = sf.read(wav_file, dtype="float32")
                    # Resample to 16kHz if needed
                    if sr != 16000:
                        from scipy import signal
                        num_samples = int(len(audio) * 16000 / sr)
                        audio = signal.resample(audio, num_samples).astype("float32")
                    self.audio_cache[wav_file] = audio
                    self._logger.info(f"Cached audio file: {wav_file}")
                except Exception as e:
                    self._logger.warning(f"Failed to cache audio file {wav_file}: {e}")
            else:
                self._logger.warning(f"Audio file not found: {wav_file}")


def generate_test_audio_files(output_dir="test_audio", num_files=15):
    """Generate test wav files for profiling"""
    os.makedirs(output_dir, exist_ok=True)
    wav_files = []
    
    for i in range(num_files):
        # Generate simple test audio (white noise, different durations)
        duration = 2 + (i % 3)  # 2-4 seconds
        sr = 16000
        audio = np.random.randn(sr * duration).astype("float32") * 0.1
        
        wav_path = os.path.join(output_dir, f"test_audio_{i:02d}.wav")
        sf.write(wav_path, audio, sr)
        wav_files.append(wav_path)
    
    return wav_files


def load_wav_files(wav_dir="test_audio", num_files=15):
    """Load wav files from directory"""
    wav_files = []
    
    if os.path.exists(wav_dir):
        # Load existing wav files
        for filename in sorted(os.listdir(wav_dir))[:num_files]:
            if filename.endswith(".wav"):
                wav_files.append(os.path.join(wav_dir, filename))
    
    # If not enough files, generate them
    if len(wav_files) < num_files:
        generated_files = generate_test_audio_files(wav_dir, num_files)
        wav_files.extend(generated_files)
    
    return wav_files[:num_files]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profiling for Moonshine Japanese Speech-to-Text")
    add_common_args(
        parser,
        model_choices=MODEL_CHOICES,
        default_model=["onnx-tiny-float"],
        default_input="cached_audio",
        input_desc="Type 'cached_audio' to use cached wav files"
    )
    
    parser.add_argument(
        "--num-audio-files",
        type=int,
        default=15,
        help="Number of cached audio files to generate/use (default: %(default)s)"
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=".cache/text_to_speech_audio",
        help="Directory to store/load test audio files (default: %(default)s)"
    )
    
    args = parser.parse_args()
    configure_logging(args.logging)

    # Load or generate cached audio files
    wav_files = load_wav_files(args.audio_dir, args.num_audio_files)
    
    if wav_files:
        print(f"\n>>> Starting profiling for Moonshine JP with {len(wav_files)} audio files...")
        
        output_file = f"profile_moonshine_jp_{datetime.date.today().strftime('%Y%m%d')}.csv"
        
        profiler = MoonshineProfiler(
            args.models,
            n_threads=args.threads
        )
        
        # Cache the audio files
        profiler.cache_audio_files(wav_files)
        
        profiler.profile_models(
            n_iters=args.repeat,
            inputs=wav_files,
            store=output_file,
            print_stats=True
        )
    else:
        print("No audio files available for profiling.")
