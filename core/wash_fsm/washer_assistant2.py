from transitions import Machine
import unicodedata
import json
from core.intent_classifier.intent_classifier import predict_washingbert1, predict_washingbert2 
from claude import chat_with_claude

SIMILARITY_THRESHOLD = 0.80  # You can adjust this value
SIMILARITY_THRESHOLD_TYPE = 0.5  # For type confidence threshold
SIMILARITY_THRESHOLD_SECOND = 0.7  # For second stage confirmations

question_ask = {
    "everyday": "どんな汚れですか？",
    "delicates": "デリケート衣類をどのように洗いますか？",
    "towels": "仕上がりの希望は？",
    "sportswear": "急いでいますか？",
    "bedding": "どのように洗いますか？",
    "whites": "強力な除菌を行いますか？",
    "care": "どのケアをしますか？",
    "dry_only": "やわらか乾燥にしますか？"
}

maintenance_json = {
    "clean_tub": "洗濯槽クリーニング",
    "drain": "排水クリーニング",
    "filter": "フィルター掃除",
    "tub_mold": "黒カビ除去",
    "low_clean_tub": "低水位槽洗浄",
    "remove_moisture": "水分除去",
    "light_rinse": "軽いすすぎ"
}

WASH_CYCLES = {
    "everyday": {"second_type": {
        "auto": "auto",
        "none_in_particular": "auto",
        "yellowing": "soak_40",
        "remove_yellowing": "soak_40",
        "sebum_sweat": "auto_40",
        "target_sebum_sweat": "auto_40",
        "mud_food_spills_strong_odor": "odor_40",
        "preferred_night": "night",
        "preferred_energy": "energy_saver",
        "preferred_powerful": "powerfull_waterfall"
    }},
    "delicates": {"second_type": {
        "gentle_wash": "delicates_30",
        "target_sebum_sweat": "delicates_40",
        "sebum_sweat": "delicates_40",
        "mud_food_spills_strong_odor": "delicates_40",
        "remove_yellowing": "delicates_40_soak",
        "yellowing": "delicates_40_soak",
        "light_refresh_overall": "home_clean"
    }},
    "towels": {"second_type": {
        "fluffy_soft": "towel_mode",
        "in_a_hurry": "towel_quick"
    }},
    "sportswear": {"second_type": {
        "in_a_hurry": "synthetics_60min",
        "not_in_a_hurry": "energy_saver"
    }},
    "bedding": {"second_type": {
        "blanket": "blanket",
        "warm_water_thorough": "blanket_40"
    }},
    "whites": {"second_type": {
        "gentle": "auto_40_whites",
        "sanitize": "auto_60_sanitize"
    }},
    "care": {"second_type": {
        "dewrinkle_deodorize": "dewrinkle",
        "pollen_sanitize_deodorize": "hygiene_care",
        "restore_water_repellency": "water_repellency_restore"
    }},
    "dry_only": {"second_type": {
        "normal": "normal_dry",
        "soft": "soft_dry"
    }}
}

# Japanese label mapping for natural output
JAPANESE_LABELS = {
    "everyday": "普段着",
    "delicates": "おしゃれ着",
    "towels": "タオル",
    "sportswear": "スポーツウェア",
    "bedding": "寝具",
    "whites": "白物",
    "care": "衣類のケア",
    "dry_only": "乾燥のみ",
    # second_type
    "auto": "自動モード",
    "none_in_particular": "指定なし",
    "yellowing": "黄ばみ落とし",
    "sebum_sweat": "皮脂・汗汚れ落とし",
    "mud_food_spills_strong_odor": "泥・食べ物・におい落とし",
    "preferred_night": "ナイトモード",
    "preferred_energy": "省エネモード",
    "preferred_powerful": "パワフルモード",
    "gentle_wash": "やさしめ",
    "target_sebum_sweat": "皮脂・汗汚れ重視",
    "remove_yellowing": "黄ばみ重視",
    "light_refresh_overall": "優しく乾燥",
    "fluffy_soft": "ふんわりやわらか",
    "in_a_hurry": "短時間モード",
    "not_in_a_hurry": "通常モード",
    "blanket": "毛布モード",
    "warm_water_thorough": "温水でしっかり",
    "gentle": "やさしく除菌",
    "sanitize": "強力除菌",
    "dewrinkle_deodorize": "シワ取り・消臭",
    "pollen_sanitize_deodorize": "花粉・除菌・消臭",
    "restore_water_repellency": "撥水回復",
    "normal": "通常乾燥",
    "soft": "やわらか乾燥"
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

class WashAssistantFSM:
    states = [
        "idle",
        "wait_second_input",
        "wait_confirm_second",
        "wait_maintenance_type",
        "complete"
    ]

    def __init__(self, text_agent, stt_agent, tts_agent):
        self.text_agent = text_agent
        self.stt_agent = stt_agent
        self.tts_agent = tts_agent
        self.intent = None
        self.type = None
        self.second_type = None
        self.intent_conf = None
        self.type_conf = None
        self.second_type_conf = None

        # Initialize FSM
        self.machine = Machine(model=self, states=self.states, initial="idle", auto_transitions=False)

        # FSM transitions
        self.machine.add_transition("got_wash_intent", "idle", "wait_second_input", after="ask_second_type")
        self.machine.add_transition("got_second_type", "wait_second_input", "wait_confirm_second", after="confirm_second_type")
        #self.machine.add_transition("select_course", "wait_second_input", "wait_confirm_second", after="announce_start")
        self.machine.add_transition("confirm_yes", "wait_confirm_second", "complete", after="announce_start")
        self.machine.add_transition("confirm_no", "wait_confirm_second", "idle", after="ask_type_again")
        self.machine.add_transition("maintenance_needed", "idle", "wait_maintenance_type", after="ask_maintenance_type")
        self.machine.add_transition("got_maintenance_type", "wait_maintenance_type", "complete", after="announce_maintenance")
        self.machine.add_transition("go_back", "*", "idle", after="handle_go_back")
        self.machine.add_transition("do_reset", "*", "idle")
        #self.machine.add_transition("cancel", "*", "idle", after="handle_cancel")

    def _speak(self, msg: str):
        wav = self.tts_agent.synthesize(msg)
        self.stt_agent.audio_manager.play(wav)
        print(f"{msg}")

    def run_stage1(self, text: str):
        return predict_washingbert1(text)

    def run_stage2(self, text: str):
        return predict_washingbert2(text)

    def resolve_command_label(self):
        if not self.type or self.type not in WASH_CYCLES:
            return None
        second_map = WASH_CYCLES[self.type]["second_type"]
        if self.second_type in second_map:
            label = second_map[self.second_type]
            if label == "none":
                return None
            return label
        return None

    def ask_second_type(self):
        
        if self.type == "whites":
            print("Announce course for whites")
            self.second_type = "sanitize"
            self.got_second_type()
        elif self.type == "dry_only":
            print("Announce course for dry only")
            self.second_type = "soft"
            self.got_second_type()
        else:
            print(f"{self.type}の洗い方を教えてください。")
            q = question_ask.get(self.type, f"{self.type}の洗い方を教えてください。")
            self._speak(q)

    def confirm_second_type(self):
        t_jp = JAPANESE_LABELS.get(self.type, self.type)
        s_jp = JAPANESE_LABELS.get(self.second_type, self.second_type)
        self._speak(f"{t_jp}を{s_jp}で洗います。よろしいですか？")

    def ask_type_again(self):
        self._speak("何を洗濯しますか？")
        self.reset()

    def ask_maintenance_type(self):
        self._speak("メンテナンスですね。どの作業をしますか？（ドラムの乾燥/洗浄/カビ取り）")

    def announce_maintenance(self):
        cycle = maintenance_json[self.type]
        self._speak(f"{cycle}を開始します。")
        self.reset()

    def announce_start(self):
        
        type_jp = JAPANESE_LABELS.get(self.type, self.type)
        second_type_jp = JAPANESE_LABELS.get(self.second_type, self.second_type)
        course_jp = JAPANESE_COURSE_LABELS.get(self.resolve_command_label(), self.resolve_command_label())
        msg = f"{type_jp}を{second_type_jp}で{course_jp}を開始します。"
        self._speak(msg)
        self.reset()

    def handle_go_back(self):
        
        self._speak("最初に戻ります。何を洗濯しますか？")
        self.reset()

    def reset(self):
        if self.state != "idle":
            self.do_reset()
        self.intent = self.type = self.second_type = None
        self.intent_conf = self.type_conf = self.second_type_conf = None
        print("FSM Reset to idle")

    def process_query(self, query):
        if not query:
            self._speak("入力が空です。もう一度お願いします。")
            return None

        if self.state == "idle":
            out1 = self.run_stage1(query)
            if not out1:
                self._speak("初回認識に失敗しました。もう一度お願いします。")
                return None
            self.intent = out1.get("primary_intent")
            self.type = out1.get("type")
            self.intent_conf = out1.get("intent_confidence")
            self.type_conf = out1.get("types_confidence")
            print(f"---INFO--- Stage1 intent={self.intent}({self.intent_conf:.3f}) type={self.type}({self.type_conf:.3f})")

            # Handle maintenance intent
            if self.intent == "maintenance":
                if self.type_conf and self.type_conf > SIMILARITY_THRESHOLD_TYPE and self.type in maintenance_json:
                    cycle = maintenance_json[self.type]
                    self._speak(f"{cycle}を開始します。")
                    self.reset()
                    return None
                else:
                    #self._speak("メンテナンスですね。どの作業をしますか？（ドラムの乾燥/洗浄/カビ取り）")
                    self.maintenance_needed()
                    #self.stage = "WAIT_MAINTENANCE_TYPE"
                    return None

            # Handle general_info intent (embedding QA only, no model)
            if self.intent == "general_info" or not self.intent_conf or self.intent_conf < SIMILARITY_THRESHOLD:
                result = self.text_agent.answer_query(query)[0]
                answer = result["answer"]
                similarity = result["similarity"]
                if similarity < SIMILARITY_THRESHOLD:
                    pass
                    # claude_resp = chat_with_claude(query)
                    # self._speak(claude_resp)
                else:
                    self._speak(answer)
                self.reset()
                return None

            # Only proceed if both intent and type confidence are above threshold for wash intent
            if self.intent_conf < SIMILARITY_THRESHOLD or self.type_conf < SIMILARITY_THRESHOLD_TYPE:
                self._speak("もう一度お願いします。")
                return None

            # if second type is "none" for wash intent
            if self.type == "none" or not self.type:
                self._speak("オプションの特定ができませんでした。もう一度詳細をお願いします。")
                return None
            
            # Ask for second_type (option) directly
            #self.stage = self.STAGE_WAIT_SECOND_INPUT
            self.got_wash_intent()
            return None


        elif self.state == "wait_second_input":
            
            out2 = self.run_stage2(query)
            if not out2:
                self._speak("詳細認識に失敗しました。もう一度お願いします。")
                return None
            second_type = out2.get("second_type")
            self.second_type_conf = out2.get("second_types_confidence")
            # If user says "back", reset to initial type selection
            if second_type == "back":
                self.go_back()
                return None
            self.second_type = second_type
            print(f"---INFO--- Stage2 second_type={self.second_type} conf={self.second_type_conf:.3f}")
            valid_second_types = set(WASH_CYCLES.get(self.type, {}).get("second_type", {}).keys())
            if second_type not in valid_second_types:
                self._speak(f"{self.type} に '{second_type}' のオプションは存在しません。もう一度選択してください。")
                print(f"Invalid combination: type={self.type}, second_type={second_type}")
                return None
            #self.stage = self.STAGE_WAIT_CONFIRM_SECOND
            self.got_second_type()
            return None

        elif self.state == "wait_confirm_second":
            out2 = self.run_stage2(query)
            if not out2:
                self._speak("はい/いいえの認識に失敗しました。もう一度お試しください。")
                return None
            result = out2.get("second_type")
            if result in {"yes"}:
                command_label = self.resolve_command_label()
                if not command_label:
                    self._speak("洗濯コースを特定できませんでした。最初からやり直してください。")
                    
                    self.reset()
                    return None
                self.confirm_yes()
            elif result in "no":
                
                self.confirm_no()
            elif result in  "back":
                self.go_back()
            else:
                self._speak("はい / いいえ で答えてください。")
            return None

        elif self.state == "wait_maintenance_type":
            out1 = self.run_stage1(query)
            if not out1:
                self._speak("認識に失敗しました。もう一度お願いします。")
                return None
            self.type = out1.get("type")
            self.type_conf = out1.get("types_confidence")
            if self.type_conf and self.type_conf > SIMILARITY_THRESHOLD_TYPE and self.type in maintenance_json:
                self.got_maintenance_type()
            else:
                self._speak("もう一度、どのメンテナンス作業か教えてください。")
            return None