import os
import time
import asyncio
import argparse
from tsuki_onnx import Tsuki
from typing import Final
import soundfile as sf
from .base import BaseTextToSpeechModel

DEFAULT_TSUKI_VOICE: Final[str] = "fumi_f_ja"

class TsukiTextToSpeech(BaseTextToSpeechModel):
    def __init__(self, voice: str | None = DEFAULT_TSUKI_VOICE, model_dir: str | os.PathLike | None = None):
        super().__init__(voice)
        self.model_dir = model_dir
        if not model_dir:
            # Use dynamic absolute path for model_dir
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) )
            #model_dir = os.path.join(base_dir, "models", voice)
            model_dir = os.path.join(base_dir, "jp-models", voice)
            self.model_dir = model_dir
        # self._model = Tsuki(model_dir)
        
    def _load_model(self,model_dir):
        self._model = Tsuki(model_dir)

    def _unload_model(self):
        self._model = None

    def synthesize_to_wav(self, text: str, output_filename: str | os.PathLike):
        self._load_model(self.model_dir)
        audio, sr = self._model.generate(text)
        sf.write(output_filename, audio, sr)
        self._unload_model()
    def synthesize_stream(self, text: str, speed: float = 1.0, **kwargs):
        yield from self._model.generate_stream(text)

    async def synthesize_stream_async(self, text: str, speed: float = 1.0, **kwargs):
        async for audio, sr in self._model.generate_stream(text):
            yield audio, sr

async def generate_async(model, text, audio_manager, idx):
    chunk_idx = 0
    chunk_files = []
    async for audio, sr in model.generate_stream(text):
        chunk_filename = f"speech-stream-chunk-{chunk_idx}.wav"
        sf.write(chunk_filename, audio, sr)
        # print(f"Saved: {chunk_filename}")
        audio_manager.play(chunk_filename)
        chunk_files.append(chunk_filename)
        chunk_idx += 1
    for fname in chunk_files:
        try:
            os.remove(fname)
        except Exception:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tsuki Text-to-Speech command line tool.")
    parser.add_argument("text", type=str, help="Text to convert to speech")
    parser.add_argument("-o", "--output", type=str, default="output.wav", help="Output WAV file (default: %(default)s)")
    parser.add_argument("-v", "--voice", type=str, default=DEFAULT_TSUKI_VOICE, help="Voice/model name (default: %(default)s)")
    parser.add_argument("-m", "--model_dir", type=str, default=None, help="Model directory (default: ../models/<voice>)")
    args = parser.parse_args()

    tts = TsukiTextToSpeech(voice=args.voice, model_dir=args.model_dir)
    tts.synthesize_to_wav(args.text, args.output)
    print(f"Audio written to: {args.output}")

    model_dir = "../models/fumi_f_ja"
    texts = [
"金融機関や大企業などが集中し、新聞・放送・出版などの文化面、大学・研究機関などの教育・学術面においても日本の中枢をなす。交通面でも鉄道網 道路網、航空路の中心である。 "
    ]

    t0 = time.time()
    model = Tsuki(model_dir)
    load_t = time.time() - t0
    print(f"Loaded model {os.path.basename(model_dir)} in {load_t:.2f}s")
    print("")

    audio_manager = AudioManager()  # Instantiate AudioManager

    total_start = time.time()  # Start total timer
    for idx, text in enumerate(texts):
        print(text)
        asyncio.run(generate_async(model, text, audio_manager, idx))
        print("")
    total_elapsed = time.time() - total_start
    print(f"Total time for all generations: {total_elapsed:.2f}s")
