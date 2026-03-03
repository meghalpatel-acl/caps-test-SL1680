import json
from pathlib import Path
from typing import Final
import time

from tqdm import tqdm

from core.text_to_speech import TextToSpeechAgent
from core.utils.download import download_from_url, download_from_hf
from config import DEFAULT_QA_FILE

DATA_DIR: Final = Path("./data")
MODELS_DIR: Final = Path("./models")

JAPANESE_LABELS = {
    "everyday": {"普段着": {
        "auto": "自動モード",
        "none_in_particular": "指定なし",
        "yellowing": "黄ばみ落とし",
        "sebum_sweat": "皮脂・汗汚れ落とし",
        "mud_food_spills_strong_odor": "泥・食べ物・におい落とし",
        "preferred_night": "ナイトモード",
        "preferred_energy": "省エネモード",
        "preferred_powerful": "パワフルモード"
    }},
    "delicates": {"おしゃれ着":{
        "gentle_wash": "やさしめ",
        "target_sebum_sweat": "皮脂・汗汚れ重視",
        "remove_yellowing": "黄ばみ重視",
        "light_refresh_overall": "優しく乾燥"
    }},
    "towels": {"タオル":{
        "fluffy_soft": "ふんわりやわらか",
        "in_a_hurry": "短時間モード"
    }},
    "sportswear": {"スポーツウェア":{
        "in_a_hurry": "短時間モード",
        "not_in_a_hurry": "通常モード"
    }},
    "bedding": {"寝具":{
        "blanket": "毛布モード",
        "warm_water_thorough": "温水でしっかり"
    }},
    "whites": {"白物":{
        "sanitize": "強力除菌",
    }},
    "care": {"衣類のケア":{
        "dewrinkle_deodorize": "シワ取り・消臭",
        "pollen_sanitize_deodorize": "花粉・除菌・消臭",
        "restore_water_repellency": "撥水回復",
    }},
    "dry_only": {"乾燥のみ":{
        "soft": "やわらか乾燥"
    }}
}

WASH_CYCLE_MAP = {
    "everyday": {
        "auto": "auto",
        "none_in_particular": "auto",
        "yellowing": "soak_40",
        "sebum_sweat": "odor_40",
        "mud_food_spills_strong_odor": "powerfull_waterfall",
        "preferred_night": "night",
        "preferred_energy": "energy_saver",
        "preferred_powerful": "powerfull_waterfall",
    },
    "delicates": {
        "gentle_wash": "delicates_30",
        "target_sebum_sweat": "delicates_40",
        "remove_yellowing": "delicates_40_soak",
        "light_refresh_overall": "home_clean",
    },
    "towels": {
        "fluffy_soft": "towel_mode",
        "in_a_hurry": "towel_quick",
    },
    "sportswear": {
        "in_a_hurry": "synthetics_60min",
        "not_in_a_hurry": "auto",
    },
    "bedding": {
        "blanket": "blanket",
        "warm_water_thorough": "blanket_40",
    },
    "whites": {
        "sanitize": "auto_60_sanitize",
    },
    "care": {
        "dewrinkle_deodorize": "dewrinkle",
        "pollen_sanitize_deodorize": "hygiene_care",
        "restore_water_repellency": "water_repellency_restore",
    },
    "dry_only": {
        "soft": "soft_dry",
    },
}

JAPANESE_COURSE_LABELS = {
    "auto": "自動コース",
    "soak_40": "つけおき40℃コース",
    "auto_40": "自動40℃コース",
    "odor_40": "におい除去40℃コース",
    "night": "夜静かコース",
    "energy_saver": "省エネコース",
    "powerfull_waterfall": "パワフル滝洗いコース",
    "delicates_30": "デリケート30℃コース",
    "delicates_40": "デリケート40℃コース",
    "delicates_40_soak": "デリケート40℃つけおきコース",
    "home_clean": "ホームリフレッシュコース",
    "towel_mode": "タオルふんわりコース",
    "towel_quick": "タオル急ぎコース",
    "synthetics_60min": "合成繊維60分コース",
    "blanket": "毛布コース",
    "blanket_40": "毛布温水コース",
    "auto_40_whites": "白物40℃コース",
    "auto_60_sanitize": "白物60℃除菌コース",
    "dewrinkle": "シワ取りコース",
    "hygiene_care": "衛生ケアコース",
    "water_repellency_restore": "撥水回復コース",
    "normal_dry": "通常乾燥コース",
    "soft_dry": "やわらか乾燥コース"
}


responses_gen = {
    #Followup question
    "everyday": "どんな汚れですか？",
    "delicates": "デリケート衣類をどのように洗いますか？",
    "towels": "仕上がりの希望は？",
    "sportswear": "急いでいますか？",
    "bedding": "どのように洗いますか？",
    "whites": "強力な除菌を行いますか？",
    "care": "どのケアをしますか？",
    "dry_only": "やわらか乾燥にしますか？",
    #Maintenance cycle selection
    "clean_tub": "洗濯槽クリーニングを開始します。",
    "tub_mold": "黒カビ除去を開始します。",
    "low_clean_tub": "低水位槽洗浄を開始します。",
    "remove_moisture": "水分除去を開始します。",
    "light_rinse": "軽いすすぎを開始します。",
    #Ask question
    "ask_type_again": "何を洗濯しますか？",
    "ask_maintenance_type": "メンテナンスですね。どの作業をしますか？（ドラムの乾燥/洗浄/カビ取り）",
    "go_back": "最初に戻ります。何を洗濯しますか？",
    "not_query": "入力が空です。もう一度お願いします。",
    "not_first_query": "初回認識に失敗しました。もう一度お願いします。",
    "low_conf": "もう一度お願いします。",
    "type_none": "オプションの特定ができませんでした。もう一度詳細をお願いします。",
    "not_second_query": "詳細認識に失敗しました。もう一度お願いします。",
    "not_confirm_query": "はい/いいえの認識に失敗しました。もう一度お試しください。",
    "not_cmd_label": "洗濯コースを特定できませんでした。最初からやり直してください。",
    "not_yes_no_back": "はい / いいえ で答えてください。",
    "not_maintenance_type": "認識に失敗しました。もう一度お願いします。",
    "low_conf_maintenance_type": "もう一度、どのメンテナンス作業か教えてください。",
}



def generate_all_japanese_speak_sentences(labels_dict):
    results = []

    for type_key, type_block in labels_dict.items():
        # type_block = {"普段着": {...}}
        jp_type = next(iter(type_block.keys()))
        second_map = type_block[jp_type]

        for second_key, jp_second in second_map.items():
            sentence = f"{jp_type}を{jp_second}で洗います。よろしいですか？"

            results.append(sentence)

    return results


def generate_all_announcements(jp_labels, course_labels, cycle_map):
    results = []

    for type_key, type_dict in jp_labels.items():
        # type_dict has exactly one JP key
        type_jp, second_types = next(iter(type_dict.items()))

        for second_key, second_jp in second_types.items():
            course_key = cycle_map[type_key][second_key]
            course_jp = course_labels[course_key]

            msg = f"{type_jp}を{second_jp}で{course_jp}を開始します。"

            results.append(msg)

    return results



if __name__ == "__main__":
    YELLOW: Final = "\033[93m"
    GREEN: Final = "\033[32m"
    CYAN: Final = "\033[36m"
    RESET: Final = "\033[0m"

    responses = []
    print(CYAN + "Downloading models..." + RESET)
    # download MiniLM models
    # download_from_url(
    #     url="https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/all-MiniLM-L6-v2-Q8_0.gguf",
    #     filename=MODELS_DIR / f"gguf/all-MiniLM-L6-v2-Q8_0.gguf"
    # )
    # download_from_url(
    #     url="https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/all-MiniLM-L6-v2.synap",
    #     filename=MODELS_DIR / f"synap/all-MiniLM-L6-v2.synap"
    # )
    # download granite embedding model to models/gguf/
    # download_from_hf(
    #     repo_id="bartowski/granite-embedding-107m-multilingual-GGUF",
    #     filename="granite-embedding-107m-multilingual-Q8_0.gguf"
    # )
    # Optionally, move the file to models/gguf/ if huggingface_hub puts it elsewhere
    import shutil, os
    src = "models/bartowski/granite-embedding-107m-multilingual-GGUF/granite-embedding-107m-multilingual-Q8_0.gguf"
    dst = "models/gguf/granite-embedding-107m-multilingual-Q8_0.gguf"
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
    # download Moonshine models
    # download_from_hf(
    #     repo_id="UsefulSensors/moonshine",
    #     filename=f"onnx/merged/tiny/float/encoder_model.onnx",
    # )
    # download_from_hf(
    #     repo_id="UsefulSensors/moonshine",
    #     filename=f"onnx/merged/tiny/float/decoder_model_merged.onnx",
    # )
    # download_from_hf(
    #     repo_id="UsefulSensors/moonshine-base", 
    #     filename="config.json"
    # )
    # download_from_hf(
    #     repo_id="UsefulSensors/moonshine-base", 
    #     filename="tokenizer.json"
    # )
    # download_from_url(
    #     url="https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/moonshine_tiny_float_encoder.synap",
    #     filename=MODELS_DIR / f"synap/moonshine/tiny/float/encoder.synap"
    # )
    # download_from_url(
    #     url="https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/moonshine_tiny_float_decoder_uncached.synap",
    #     filename=MODELS_DIR / f"synap/moonshine/tiny/float/decoder_uncached.synap"
    # )
    # download_from_url(
    #     url="https://github.com/spal-synaptics/on-device-assistant/releases/download/models-v1/moonshine_tiny_float_decoder_cached.synap",
    #     filename=MODELS_DIR / f"synap/moonshine/tiny/float/decoder_cached.synap"
    # )
    # download piper-tts models
    # download_from_hf(
    #     repo_id="rhasspy/piper-voices",
    #     filename="en/en_US/lessac/low/en_US-lessac-low.onnx"
    # )
    # download_from_hf(
    #     repo_id="rhasspy/piper-voices",
    #     filename="en/en_US/lessac/low/en_US-lessac-low.onnx.json"
    # )

    spoken_list = generate_all_japanese_speak_sentences(JAPANESE_LABELS)

    for item in spoken_list:
        print(item)

    announcements = generate_all_announcements(
        JAPANESE_LABELS,
        JAPANESE_COURSE_LABELS,
        WASH_CYCLE_MAP
    )

    # for item in announcements:
    #     print(item)

    print(GREEN + "Downloads complete." + RESET)

    print(CYAN + "Generating TTS cache for answers in" + f" {DEFAULT_QA_FILE}..." + RESET)
    # Generate TTS cache for answers in DEFAULT_QA_FILE
    tts = TextToSpeechAgent()
    qa_file = Path(DEFAULT_QA_FILE)
    if qa_file.exists():
        with open(qa_file, "r") as f:
            answers = [pair["answer"] for pair in json.load(f)]
            for item in announcements:
                answers.append(item)
            for item in spoken_list:
                answers.append(item)
            for answer in tqdm(answers, desc=qa_file.name):
                tts.synthesize(answer)
                time.sleep(0.5)  # Sleep for 500ms between syntheses
        print(GREEN + f"TTS cache generation complete for {qa_file.name}." + RESET)
    else:
        print(YELLOW + f"QA file {qa_file} not found, skipping TTS cache generation." + RESET)
