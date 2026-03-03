import os
import hashlib
import logging
from typing import Final, Literal, Any
import time
import soundfile as sf
from ..utils.audio import AudioManager
from config import DEFAULT_QA_FILE

MODEL_CHOICES: Final[list] = ["piper", "kitten",  "tsuki"]

logger = logging.getLogger(__name__)


def tts_factory(model: Literal["piper", "kitten","tsuki"], voice: str | None):
    
    if model == "tsuki":
        try:
            from .tsuki import TsukiTextToSpeech, DEFAULT_TSUKI_VOICE
            return TsukiTextToSpeech(voice or DEFAULT_TSUKI_VOICE)
        except ImportError:
            logger.error("Tsuki TTS not available.")
            raise
    elif model == "kitten":
        try:
            from .kitten import KittenTextToSpeech, DEFAULT_KITTEN_VOICE
            return KittenTextToSpeech(voice or DEFAULT_KITTEN_VOICE)
        except ImportError:
            logger.error("Kitten TTS not available.")
            raise
    elif model == "piper":
        try:
            from .piper import PiperOnnx, DEFAULT_PIPER_VOICE
            return PiperOnnx(voice or DEFAULT_PIPER_VOICE)
        except ImportError:
            logger.error("Piper TTS not available.")
            raise
    else:
        raise ValueError(f"Invalid model '{model}', please use one of {MODEL_CHOICES}")



# Example usage:
# precache_all_answers()  # Uncomment this line to run caching at startup


class TextToSpeechAgent:

    def __init__(
        self,
        tts_model: Literal["piper", "kitten", "tsuki"] = "tsuki",
        tts_voice: str | None = None,
        output_dir: str = ".cache/text_to_speech",
        audio_manager: AudioManager | None = None,
        stt_manager: Any = None  # Add optional stt_manager
    ):
        self.tts = tts_factory(tts_model, tts_voice)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.audio_manager = audio_manager or AudioManager()
        self.stt_manager = stt_manager  # Store stt_manager

    @staticmethod
    def file_checksum(content: str, hash_length: int = 16) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:hash_length]

    def synthesize(self, text: str, output_filename: str = None, play_audio: bool = False) -> str:
        if output_filename is None:
            chk = self.file_checksum(self.tts.voice + text)
            output_filename = os.path.join(self.output_dir, f"speech-output-{chk}.wav")

        if os.path.exists(output_filename):
            logger.debug("Found TTS cache at '%s'", output_filename)
        else:
            self.tts.synthesize_to_wav(text, output_filename)
            logger.debug("Cached TTS to '%s'", output_filename)

        if play_audio:
            # Pause STT before playback if stt_manager is available
            if self.stt_manager and hasattr(self.stt_manager, "pause"):
                self.stt_manager.pause()
            if self.audio_manager.device:
                self.audio_manager.play(output_filename)
            else:
                logger.warning("Skipping audio playback, no valid playback device in audio manager")
            # Resume STT after playback if stt_manager is available
            if self.stt_manager and hasattr(self.stt_manager, "resume"):
                self.stt_manager.resume()
        return output_filename

    def synthesize_stream(self, text: str, speed: float = 1.0, **kwargs):
        """Yield audio chunks (samples, sample_rate) as they are generated."""
        yield from self.tts.synthesize_stream(text, speed=speed, **kwargs)

    async def synthesize_stream_async(self, text: str, audio_manager=None, output_filename="output.wav", speed: float = 1.0, **kwargs):
        t0 = time.time()
        chunk_files = []
        chunk_idx = 0
        synth_time = 0
        playback_time = 0
        async for audio, sr in self.tts.synthesize_stream_async(text, speed=speed, **kwargs):
            gen_t = time.time() - t0
            synth_time += gen_t
            chunk_filename = os.path.join(self.output_dir, f"speech-stream-chunk-{chunk_idx}.wav")
            sf.write(chunk_filename, audio, sr)
            # print(f"Saved: {chunk_filename}")
            chunk_files.append(chunk_filename)
            play_start = time.time()
            if audio_manager and hasattr(audio_manager, "play"):
                audio_manager.play(chunk_filename)
            playback_time += time.time() - play_start
            chunk_idx += 1
            t0 = time.time()
        total_time = synth_time + playback_time
        print(f"TTS stats: synthesis={synth_time*1000:.3f}ms, playback={playback_time*1000:.3f}ms, total={total_time*1000:.3f}ms")
        # Delete all chunk files
        for fname in chunk_files:
            try:
                os.remove(fname)
            except Exception as e:
                print(f"Warning: could not delete {fname}: {e}")
        return output_filename