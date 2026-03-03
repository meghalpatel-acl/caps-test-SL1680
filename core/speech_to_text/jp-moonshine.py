import time
from pathlib import Path

import numpy as np
import onnxruntime
from tokenizers import Tokenizer

from .base import BaseSpeechToTextModel

LOCAL_MODEL_DIR = Path("jp-models/moonshine_jp")

class JPMoonshineOnnx(BaseSpeechToTextModel):
    def __init__(self, rate: int = 16_000, n_threads: int | None = None):
        # Set up paths to local files
        config_path = LOCAL_MODEL_DIR / "config.json"
        tokenizer_path = LOCAL_MODEL_DIR / "tokenizer.json"
        encoder_path = LOCAL_MODEL_DIR / "encoder_model.onnx"
        decoder_path = LOCAL_MODEL_DIR / "decoder_model_merged.onnx"

        # Load config and tokenizer directly from local files
        self.config = self._load_config(config_path)
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.eos_token_id = self.config["eos_token_id"]
        self.decoder_start_token_id = self.config["decoder_start_token_id"]
        self.num_key_value_heads = self.config["decoder_num_key_value_heads"]
        self.dim_kv = (
            self.config["hidden_size"] // self.config["decoder_num_attention_heads"]
        )
        self.decoder_layers = self.config["decoder_num_hidden_layers"]
        self.max_len = self.config["max_position_embeddings"]
        self.rate = rate

        opts = onnxruntime.SessionOptions()
        if n_threads is not None:
            opts.intra_op_num_threads = n_threads
            opts.inter_op_num_threads = n_threads
        self.encoder_session = onnxruntime.InferenceSession(str(encoder_path), opts, providers=['CPUExecutionProvider'])
        self.decoder_session = onnxruntime.InferenceSession(str(decoder_path), opts, providers=['CPUExecutionProvider'])

        self._transcribe_times = []
        self._infer_stats = {}

    def _load_config(self, path: Path):
        import json
        with open(path, "r") as f:
            return json.load(f)

    def _generate(self, audio: np.ndarray, max_len: int | None = None) -> np.ndarray:
        if max_len is None:
            max_len = min((audio.shape[-1] // self.rate) * 13, self.max_len)
        self._infer_stats["input_size"] = audio.shape[-1]
        enc_st = time.time()
        enc_out = self.encoder_session.run(None, {"input_values": audio})[0]
        enc_et = time.time()
        self._infer_stats["encoder_infer_time_ms"] = (enc_et - enc_st) * 1000

        batch_size = enc_out.shape[0]
        input_ids = np.array(
            [[self.decoder_start_token_id]] * batch_size, dtype=np.int64
        )
        past_kv = {
            f"past_key_values.{layer}.{mod}.{kv}": np.zeros(
                [batch_size, self.num_key_value_heads, 0, self.dim_kv], dtype=np.float32
            )
            for layer in range(self.decoder_layers)
            for mod in ("decoder", "encoder")
            for kv in ("key", "value")
        }
        gen_tokens = input_ids

        dec_st = time.time()
        for i in range(max_len):
            use_cache_branch = i > 0
            dec_inputs = {
                "input_ids": gen_tokens[:, -1:],
                "encoder_hidden_states": enc_out,
                "use_cache_branch": [use_cache_branch],
                **past_kv,
            }
            out = self.decoder_session.run(None, dec_inputs)
            logits = out[0]
            present_kv = out[1:]
            next_tokens = logits[:, -1].argmax(axis=-1, keepdims=True)
            for j, key in enumerate(past_kv):
                if not use_cache_branch or "decoder" in key:
                    past_kv[key] = present_kv[j]
            gen_tokens = np.concatenate([gen_tokens, next_tokens], axis=-1)
            if (next_tokens == self.eos_token_id).all():
                break
        dec_et = time.time()
        self._infer_stats["decoder_infer_time_ms"] = (dec_et - dec_st) * 1000
        self._infer_stats["decoder_tokens"] = i
        return gen_tokens

    def transcribe(self, speech: np.ndarray) -> str:
        self._infer_stats = {}
        st = time.time()
        speech = speech.astype(np.float32)[np.newaxis, :]
        tokens = self._generate(speech)
        text = self.tokenizer.decode_batch(tokens, skip_special_tokens=True)[0]
        et = time.time()
        self._transcribe_times.append(et - st)
        return text
