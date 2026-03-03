import argparse
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Final
import asyncio

from core.embeddings import TextEmbeddingsAgent
from core.wash_fsm.washer_assistant import WashAssistantFSM
from core.speech_to_text import SpeechToTextAgent, STT_MODEL_SIZES, STT_QUANT_TYPES
from core.text_to_speech import TextToSpeechAgent, MODEL_CHOICES as TTS_MODELS
#from claude import chat_with_claude

import os
from config import DEFAULT_QA_FILE

DEFAULT_SPEECH_THRESH: Final = 0.5
DEFAULT_SILENCE_DUR_MS: Final = 600 
# Get the absolute path of the current script
script_path = os.path.dirname(__file__)

def configure_logging(verbosity: str):
    level = getattr(logging, verbosity.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {verbosity}")

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


def run_command(command: str):
    try:
        out = subprocess.check_output(command, shell=True).decode().strip()
    except Exception as e:
        out = f"[error: {e}]"
    return out


def replace_tool_tokens(answer: str, tools: dict[str, str]):
    for tool in tools:
        token = tool["token"]
        if token in answer:
            output = run_command(tool["command"])
            answer = answer.replace(token, output)
    return answer


def main():
    YELLOW: Final = "\033[93m"
    GREEN: Final = "\033[32m"
    RESET: Final = "\033[0m"
    #SIMILARITY_THRESHOLD = 0.80 # You can adjust this value
    
    # Initialize conversation history for Claude
    #claude_conversation_history = []
    #MAX_HISTORY_LENGTH = 10  # Limit history to last 10 interactions (5 user + 5 assistant messages)

    def handle_speech_input(transcribed_text: str):
        #nonlocal claude_conversation_history
        print(YELLOW + f"Query: {transcribed_text}" + RESET + f" ({stt_agent.last_infer_time * 1000:.3f} ms)")
        val = washer_agent.process_query(transcribed_text)
        # print("Output:", val)
        """
        intent, command, tags = predict_onnx_standalone(transcribed_text)
        print(f"  Predicted Intent:  {intent}")
        print(f"  Predicted Command: {command}")
        print(f"  Predicted Tags:    {tags if tags else ('None',)}")
        """
        
        #ans = washer_agent.process(transcribed_text)
        #print("Ans:", ans)
        

        """
        result = text_agent.answer_query(transcribed_text)[0]
        answer, similarity, emb_infer_time = result["answer"], result["similarity"], result["infer_time"]
        if similarity < SIMILARITY_THRESHOLD:
            print(f"Similarity {similarity:.6f} below threshold, using Claude as fallback...")
            
            claude_response, claude_conversation_history = chat_with_claude(
                transcribed_text, 
                conversation_history=claude_conversation_history
            )
            
            # Limit conversation history to the most recent interactions
            if len(claude_conversation_history) > MAX_HISTORY_LENGTH:
                claude_conversation_history = claude_conversation_history[-MAX_HISTORY_LENGTH:]
            
            fallback_msg = claude_response
            
            print(GREEN + f"Agent (Claude): {fallback_msg}")
            
            tts_start = time.time()
            # Use synthesize_stream_async for Claude fallback
            async def play_stream():
                await tts_agent.synthesize_stream_async(fallback_msg, audio_manager=stt_agent.audio_manager)
            asyncio.run(play_stream())
            tts_synthesis_time = time.time() - tts_start
            
            # Play time is included in synthesize_stream_async stats
        else:
            print(GREEN + f"Agent Local: {answer}" + RESET + f" ({emb_infer_time * 1000:.3f} ms, Similarity: {similarity:.6f})")
            
            tts_start = time.time()
            wav_path = tts_agent.synthesize(answer)
            tts_synthesis_time = time.time() - tts_start
            
            # Play audio and track time
            play_start = time.time()
            stt_agent.audio_manager.play(wav_path)
            play_time = time.time() - play_start
            
            print(f"TTS stats: synthesis={tts_synthesis_time*1000:.3f}ms, playback={play_time*1000:.3f}ms, total={(tts_synthesis_time+play_time)*1000:.3f}ms")
            """
    text_agent = TextEmbeddingsAgent(args.qa_file, cpu_only=args.cpu_only, cpu_cores=args.threads)
    stt_agent = SpeechToTextAgent(
        args.stt_size, args.stt_quant, handle_speech_input, 
        cpu_only=args.cpu_only, 
        n_threads=args.threads,
        threshold=args.threshold,
        min_silence_duration_ms=args.silence_ms
    )
    tts_agent = TextToSpeechAgent(tts_model=args.tts_model,tts_voice=args.tts_voice)
    washer_agent = WashAssistantFSM(text_agent, stt_agent, tts_agent)
    stt_agent.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q&A AI Assistant")
    parser.add_argument(
        "--qa-file",
        type=str,
        default=DEFAULT_QA_FILE,
        help="Path to Question-Answer pairs (default: %(default)s)"
    )
    parser.add_argument(
        "--logging",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity: %(choices)s (default: %(default)s)"
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        default=True,
        help="Use CPU only models"
    )
    parser.add_argument(
        "-j", "--threads",
        type=int,
        help="Number of cores to use for CPU execution (default: all)"
    )
    ###################################
    parser.add_argument(
        "--test-mode",
        action="store_true",
        default=False,
        help="Run in test mode"
    )
    ###################################
    tts_args = parser.add_argument_group("text-to-speech options")
    tts_args.add_argument(
        "--tts-model",
        type=str,
        choices=TTS_MODELS,
        default="tsuki",
        help="Text-to-speech model (default: %(default)s)"
    )
    tts_args.add_argument(
        "--tts-voice",
        type=str,
        help="Voice for text-to-speech model"
    )
    stt_args = parser.add_argument_group("speech-to-text options")
    stt_args.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SPEECH_THRESH,
        help="Speech threshold, increase to lower mic capture sensitivity (default: %(default)s)"
    )
    stt_args.add_argument(
        "--silence-ms",
        type=int,
        default=DEFAULT_SILENCE_DUR_MS,
        help="Length of silence that determines end of speech (default: %(default)s ms)"
    )
    stt_args.add_argument(
        "--stt-size",
        type=str,
        choices=STT_MODEL_SIZES,
        default="tiny",
        help="Speech-to-text model size (default: %(default)s)"
    )
    stt_args.add_argument(
        "--stt-quant",
        type=str,
        choices=STT_QUANT_TYPES,
        default="float",
        help="Speech-to-text model quantization type (default: %(default)s)"
    )
    args = parser.parse_args()

    if args.test_mode:
        import test_mode as tm
        tm.main()
        exit()
    else:
        configure_logging(args.logging)
        logger = logging.getLogger(__name__)
        logger.info("Starting demo...")
        main()
