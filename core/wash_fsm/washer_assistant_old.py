import json
import time
from core.intent_classifier.intent_classifier import predict_onnx
import asyncio
import unicodedata
import re

from claude import chat_with_claude

SIMILARITY_THRESHOLD = 0.80 # You can adjust this value

# Follow-up questions

question_ask = {
    "everyday": "どんな汚れですか？（なし | 皮脂・汗・黄ばみ | 泥・食べ物・強いにおい | お好み）",
    "delicates": "デリケート衣類をどのように洗いますか？（やさしく洗う / 油汚れ・汗を重点的に / 黄ばみを重点的に / 軽くリフレッシュ）",
    "towels": "仕上がりの希望は？（ふんわり/やわらか / 急ぎ）",
    "sportswear": "急いでいますか？（はい/いいえ）",
    "bedding": "どちらを洗いますか？（毛布 / 掛け布団）",
    "whites": "強力な除菌を行いますか？（はい/いいえ）",
    "care": "どのケアをしますか？（シワ取り / 臭いを取る / 水をはじくようにする）",
    "dry_only": "やわらか乾燥にしますか？（はい/いいえ）"
}

# "はい。","はい？","はい。 ","はい？ ", "ええ。","ええ？","ええ ","ええ。 ","ええ？ ","うん。","うん？","うん ","うん。 ","うん？ ","もちろん。","もちろん？","もちろん ","もちろん。 ","もちろん？ ","オーケー。","オーケー？","オーケー ","オーケー。 ","オーケー？ ",,"了解。","了解？","了解 ","了解。 ","了解？ "
YES = {"はい","はい ","ええ","うん","もちろん", "オーケー", "了解"}
# "いいえ。","いいえ？","いいえ ","いいえ。 ","いいえ？ ", "ノー。","ノー？","ノー ","ノー。 ","ノー？ ","いや。","いや？","いや ","いや。 ","いや？ "
NO = {"いいえ","ノー","いや"}

# "急ぎ。","急ぎ？","急ぎ ","急ぎ。 ","急ぎ？ ","急いで。","急いで？","急いで ","急いで。 ","急いで？ ","早く。","早く？","早く ","早く。 ","早く？ ","すぐ。","すぐ？","すぐ ","すぐ。 ","すぐ？ ","至急。","至急？","至急 ","至急。 ","至急？ ","急用。","急用？","急用 ","急用。 ","急用？ ","今すぐ。","今すぐ？","今すぐ ","今すぐ。 ","今すぐ？ ","急ぎモード。","急ぎモード？","急ぎモード ","急ぎモード。 ","急ぎモード？ "
HURRY = {"急ぎ","急いで", "早く", "すぐ","至急","急用","今すぐ","急ぎモード"}
#"ふんわり。","ふんわり？","ふんわり ","ふんわり。 ","ふんわり？ ","やわらか。","やわらか？","やわらか ","やわらか。 ","やわらか？ ", "柔らかく。","柔らかく？","柔らかく ","柔らかく。 ","柔らかく？ ","ふっくら。","ふっくら？","ふっくら ","ふっくら。 ","ふっくら？ "
FLUFFY_SOFT = {"ふんわり","やわらか","柔らかく","ふっくら"}

#"皮脂。","皮脂？","皮脂 ","皮脂。 ","皮脂？ ", "油汚れ、汗を重点的に。", "油汚れ、汗を重点的に。 ", "油汚れ、汗を重点的に？", "油汚れ、汗を重点的に？ ","油汚れ・汗を重点的に。 ", "油汚れ・汗を重点的に？", "油汚れ・汗を重点的に？ ", "油汚れ・汗を重点的に。","皮脂・汗を重点洗浄。 ","汗。","汗？","汗 ","汗。 ","汗？ ","襟汚れ。","襟汚れ？","襟汚れ ","襟汚れ。 ","襟汚れ？ ","わき汗。","わき汗？","わき汗 ","わき汗。 ","わき汗？ ","くすみ。","くすみ？","くすみ ","くすみ。 ","くすみ？ "
SEBUM = {"皮脂","油汚れ、汗を重点的に", "油汚れ・汗を重点的に", "皮脂・汗を重点洗浄", "汗","襟汚れ","わき汗","くすみ"}
#"泥。","泥？","泥 ","泥。 ","泥？ ", "泥汚れ。","泥汚れ？","泥汚れ ","泥汚れ。 ","泥汚れ？ ","食べ物。","食べ物？","食べ物 ","食べ物。 ","食べ物？ ","ソース。","ソース？","ソース ","ソース。 ","ソース？ ","お茶。","お茶？","お茶 ","お茶。 ","お茶？ ","コーヒー。","コーヒー？","コーヒー ","コーヒー。 ","コーヒー？ ","こぼれ。","こぼれ？","こぼれ ","こぼれ。 ","こぼれ？ ","におい。","におい？","におい ","におい。 ","におい？ ","臭い。","臭い？","臭い ","臭い。 ","臭い？ ","強いにおい。","強いにおい？","強いにおい ","強いにおい。 ","強いにおい？ ","シミ。","シミ？","シミ ","シミ。 ","シミ？ "
MUD_ODOR = {"泥","泥汚れ","食べ物","ソース","お茶","コーヒー","こぼれ","におい","臭い","シミ", "強いにおい"}

# "リフレッシュ。","リフレッシュ？","リフレッシュ "," 軽くリフレッシュ。","軽くリフレッシュ。 ","リフレッシュ。 ","リフレッシュ？ ","軽く洗う。","軽く洗う？","軽く洗う ","軽く洗う。 ","軽く洗う？ ","軽い。","軽い？","軽い ","軽い。 ","軽い？ ","さっぱり。","さっぱり？","さっぱり ","さっぱり。 ","さっぱり？ ","さわやかに。","さわやかに？","さわやかに ","さわやかに。 ","さわやかに？ "
REFRESH = {"リフレッシュ", "軽くリフレッシュ", "軽く洗う","軽い","さっぱり","さわやかに"}
#"黄ばみ。","黄ばみ？","黄ばみ ","黄ばみ。 ","黄ばみ？ ",  "黄ばみを重点的に。 ", "黄ばみを重点的に。 ", "黄ばみを重点的に？", "黄ばみを重点的に？ ","黄変。","黄変？","黄変 ","黄変。 ","黄変？ ","黄ばみ除去。","黄ばみ除去？","黄ばみ除去 ","黄ばみ除去。 ","黄ばみ除去？ "
YELLOW = {"黄ばみ","黄ばみを重点的に","黄変","黄ばみ除去"}

# "除菌。","除菌？","除菌 ","除菌。 ","除菌？ ", "除菌洗浄。","除菌洗浄？","除菌洗浄 ","除菌洗浄。 ","除菌洗浄？ ","殺菌。","殺菌？","殺菌 ","殺菌。 ","殺菌？ ","菌。","菌？","菌 ","菌。 ","菌？ ","バクテリア。","バクテリア？","バクテリア ","バクテリア。 ","バクテリア？ ","強力除菌。","強力除菌？","強力除菌 ","強力除菌。 ","強力除菌？ "
SANITIZE_STRONG = {"除菌","除菌洗浄","殺菌","菌","バクテリア","強力除菌"}
#"撥水。","撥水？","撥水 ","撥水。 ","撥水？ ","撥水加工。","撥水加工？","撥水加工 ","撥水加工。 ","撥水加工？ ","撥水性。","撥水性？","撥水性 ","撥水性。 ","撥水性？ ","撥水を回復。","撥水を回復？","撥水を回復 ","撥水を回復。 ","撥水を回復？ ","防水。","防水？","防水 ","防水。 ","防水？ ", "水をはじくようにする。", "水をはじくようにする。 ", "水をはじくようにする？", "水をはじくようにする？ ", "撥水加工の回復。", "撥水加工の回復？","水をはじくようにする。", "水をはじくようにする？"
RESTORE_REPELLENCY = {"撥水","撥水加工", "撥水性","撥水を回復","防水","水をはじくようにする", "撥水加工の回復", "水をはじくようにする"}

#"しわ取り。","しわ取り？","シワとり。", "シワとり？", "シワとり。 ", "シワとり？ ","しわ取り ","しわ取り。 ","しわ取り？ ","しわとり？", "しわとり。", "しわとり。 ", "しわとり？ ","シワ取り。","シワ取り？","シワ取り ","シワ取り。 ","シワ取り？ ","スチーム。","スチーム？","スチーム ","スチーム。 ","スチーム？ ","しわ。","しわ？","しわ ","しわ。 ","しわ？ ","臭いを取る。 ", "臭いを取る？ ","臭いを取る。", "臭いを取る？","脱臭。","脱臭？","脱臭 ","脱臭。 ","脱臭？ ","消臭。","消臭？","消臭 ","消臭。 ","消臭？ ", "匂いを取る ", "匂いを取る。", "匂いを取る。 ", "匂いを取る？ ", "匂いを取る？","匂いをとる。", "匂いをとる。 ", "匂いをとる？", "匂いをとる？ ","においをとる。","においをとる。 ", "においをとる？", "においをとる？ "
DEWRINKLE_DEODOR = {"しわ取り","シワとり", "しわとり", "シワ取り","スチーム","しわ","臭いを取る","脱臭","消臭","匂いをとる", "においをとる"}
#"花粉。","花粉？","花粉 ","花粉。 ","花粉？ ","アレルギー。","アレルギー？","アレルギー ","アレルギー。 ","アレルギー？ ","花粉症。","花粉症？","花粉症 ","花粉症。 ","花粉症？ "
POLLEN = {"花粉","アレルギー","花粉症"}

#"やわらか乾燥。","やわらか乾燥？","やわらか乾燥 ","やわらか乾燥。 ","やわらか乾燥？ ","やさしい乾燥。","やさしい乾燥？","やさしい乾燥 ","やさしい乾燥。 ","やさしい乾燥？ ","低温乾燥。","低温乾燥？","低温乾燥 ","低温乾燥。 ","低温乾燥？ "
SOFT_DRY = {"やわらか乾燥","やさしい乾燥","低温乾燥"}
#"やさしい。","やさしい？","やさしい ","やさしい。 ","やさしい？ ", "優しく洗う。","やさしく洗う。","やさしく洗う？","やさしく洗う ","やさしく洗う。 ","やさしく洗う？ ","普通。","普通？","普通 ","普通。 ","普通？ ","標準。","標準？","標準 ","標準。 ","標準？ "
GENTLE_WASH = {"やさしい","やさしく洗う","普通","標準"}

#"静か。","静か？","静か ","静か。 ","静か？ ","静音。","静音？","静音 ","静音。 ","静音？ ","ナイト。","ナイト？","ナイト ","ナイト。 ","ナイト？ ","夜。","夜？","夜 ","夜。 ","夜？ ","夜間。","夜間？","夜間 ","夜間。 ","夜間？ ","静かなモード。","静かなモード？","静かなモード ","静かなモード。 ","静かなモード？ ","静かな。","静かな？","静かな ","静かな。 ","静かな？ "
QUIET = {"静か","静音","ナイト","夜","夜間","静かなモード","静かな"}
#"エコ。","エコ？","エコ ","エコ。 ","エコ？ ","省エネ。","省エネ？","省エネ ","省エネ。 ","省エネ？ ","節電。","節電？","節電 ","節電。 ","節電？ ","エコモード。","エコモード？","エコモード ","エコモード。 ","エコモード？ ","省電力モード。","省電力モード？","省電力モード ","省電力モード。 ","省電力モード？ ","エゴース。","エゴース？","エゴース ","エゴース。 ","エゴース？ "
ECO = {"エコ","省エネ","節電","エコモード","省電力モード", "エゴース"}
#"パワフル。","パワフル？","パワフル ","パワフル。 ","パワフル？ ","パワフル滝洗い。","パワフル滝洗い？","パワフル滝洗い ","パワフル滝洗い。 ","パワフル滝洗い？ ","滝洗い。","滝洗い？","滝洗い ","滝洗い。 ","滝洗い？ ","徹底洗浄。","徹底洗浄？","徹底洗浄 ","徹底洗浄。 ","徹底洗浄？ ","強力洗浄。","強力洗浄？","強力洗浄 ","強力洗浄。 ","強力洗浄？ ","念入り洗い。","念入り洗い？","念入り洗い ","念入り洗い。 ","念入り洗い？ ","パワフルモード。","パワフルモード？","パワフルモード ","パワフルモード。 ","パワフルモード？ ","「ル」は静か。","「ル」は静か？","「ル」は静か ","「ル」は静か。 ","「ル」は静か？ "
POWERFULL = {"パワフル","パワフル滝洗い","滝洗い","徹底洗浄","強力洗浄","念入り洗い","パワフルモード","「ル」は静か"}

#"戻る。","戻る？","戻る ","戻る。 ","戻る？ ", "もどる。","もどる？","もどる ","もどる。 ","もどる？ ","バック。","バック？","バック ","バック。 ","バック？ ","キャンセル。","キャンセル？","キャンセル ","キャンセル。 ","キャンセル？ ","やめる。","やめる？","やめる ","やめる。 ","やめる？ ","中止。","中止？","中止 ","中止。 ","中止？ ","ストップ。","ストップ？","ストップ ","ストップ。 ","ストップ？ ","戻れ。","戻れ？","戻れ ","戻れ。 ","戻れ？ "
BACK = {"戻る","もどる","バック","キャンセル","やめる","中止","ストップ","戻れ"}

maintenance_json = {"tub_mold": "tub_mold_clean_60", "remove_moisture": "tub_dry", "clean_tub": "tub_clean",
                    "light_rinse": "tub_quick_rinse", "low_clean_tub": "tub_clean_30"}

# Compact mapping JSON
wash_cycle_json = """
{
  "everyday": {
    "second_type": {
      "none_in_particular": "auto",
      "yellowing": "soak_40",
      "sebum_sweat_yellowing": "auto_40",
      "mud_food_spills_strong_odor": "odor_40",
      "preferred": "none"
    },
    "third_type": {
      "none": "none",
      "quiet_at_night": "night",
      "save_electricity": "energy_saver",
      "wash_thoroughly": "powerfull_waterfall"
    }
  },
  "delicates": {
    "second_type": {
      "none": "none",
      "gentle_wash": "delicates_30",
      "target_sebum_sweat": "delicates_40",
      "remove_yellowing": "delicates_40_soak",
      "light_refresh_overall": "home_clean"
    },
    "third_type": { "none": "none" }
  },
  "towels": {
    "second_type": {
      "none": "none",
      "fluffy_soft": "towel_mode",
      "in_a_hurry": "towel_quick"
    },
    "third_type": { "none": "none" }
  },
  "sportswear": {
    "second_type": {
      "none": "none",
      "in_a_hurry": "synthetics_60min",
      "not_in_a_hurry": "energy_saver"
    },
    "third_type": { "none": "none" }
  },
  "bedding": {
    "second_type": {
      "none": "none",
      "blanket": "blanket",
      "warm_water_thorough": "blanket_40"
    },
    "third_type": { "none": "none" }
  },
  "whites": {
    "second_type": {
      "none": "none",
      "gentle": "auto_40_whites",
      "sanitize": "auto_60_sanitize"
    },
    "third_type": { "none": "none" }
  },
  "care": {
    "second_type": {
      "none": "none",
      "dewrinkle_deodorize": "dewrinkle",
      "pollen_sanitize_deodorize": "hygiene_care",
      "restore_water_repellency": "water_repellency_restore"
    },
    "third_type": { "none": "none" }
  },
  "dry_only": {
    "second_type": {
      "none": "none",
      "normal": "normal_dry",
      "soft": "soft_dry"
    },
    "third_type": { "none": "none" }
  }
}
"""

WASH_CYCLES = json.loads(wash_cycle_json)


class WashAssistantFSM:
    def __init__(self, text_agent, stt_agent, tts_agent):
        self.ask_que = False
        self.cur_cloth_type = None
        self.second_type = None
        self.text_agent = text_agent
        self.stt_agent = stt_agent
        self.tts_agent = tts_agent

    def remove_punctuation_and_symbols(self, text: str) -> str:
        """
        Remove all punctuation, symbols, and spaces from text.
        Keeps Japanese Kana, Kanji, and alphanumerics only.
        """
        cleaned = []
        for ch in text:
            cat = unicodedata.category(ch)
            # Skip punctuation (P*), symbols (S*), and spaces/separators (Z*)
            if cat.startswith(("P", "S", "Z")):
                continue
            cleaned.append(ch)
        return "".join(cleaned)

    def preprocess_japanese_text(self, text: str) -> str:
        """
        Normalize and clean Japanese text for NLU models.
        """
        if not text:
            return ""
        text = unicodedata.normalize("NFKC", text)
        text = self.remove_punctuation_and_symbols(text)

        return text

    def get_command_label(self, wash_type, second_type, third_type):
        if wash_type not in WASH_CYCLES:
            print("wash_type not present:", wash_type)
            return None
        labels = WASH_CYCLES[wash_type]
        #print("Labels:",labels)
        if second_type in labels["second_type"]:
            if second_type == "none":
              if wash_type in question_ask:
                #print("Question:", question_ask[wash_type])
                self.cur_cloth_type = wash_type
                #print("Stored cloth type:",wash_type)
                print(f"Need more info for washing {wash_type}:", question_ask.get(wash_type, "Please clarify."))
                msg = f"洗濯に関する詳細情報が必要です {wash_type}: {question_ask.get(wash_type, 'Please clarify.')}"
                wav_path = self.tts_agent.synthesize(msg)
                self.stt_agent.audio_manager.play(wav_path)
                print(f"洗濯に関する詳細情報が必要です {wash_type}:", question_ask.get(wash_type, "Please clarify."))
                self.ask_que = True
                return "none"
              else:
                print("No question present for ", wash_type)
                return None
            else:
                if labels["second_type"][second_type] == "none":
                    #print("check if it is preferred type")
                    if second_type == "preferred":
                        #print("check in third type")
                        if third_type in labels["third_type"]:
                          #print("Ask for type of wash")
                          if third_type == "none":
                            self.cur_cloth_type = wash_type
                            self.second_type = "preferred"
                            #print("Ask for preferred wash / night/ eco/ powerful")
                            print(f"Need more info for {wash_type}: What type of preferred finish? (eco wash | quiet at night | wash thoroughly)")
                            msg = f"洗濯に関する詳細情報が必要です {wash_type}: どのような仕上がりがお好みですか？（省電力モード｜静かなモード｜パワフルモード）?"
                            wav_path = self.tts_agent.synthesize(msg)
                            self.stt_agent.audio_manager.play(wav_path)
                            print(f"洗濯に関する詳細情報が必要です {wash_type}: どのような仕上がりがお好みですか？（省電力モード｜静かなモード｜パワフルモード）?")
                            self.ask_que = True
                            #print("Question: What type of preferred finish? (eco wash | quiet at night | wash thoroughly)")
                            return "none"
                          else:
                            return labels["third_type"][third_type]
                    else:
                      print("Unknown option")
                      return None
                else:
                  return labels["second_type"][second_type]
                #if labels["second_type"][second_type] != "none":
                  #return labels["second_type"][second_type]
          #if third_type in labels["third_type"]:
              #if labels["third_type"][third_type] != "none":
                  #return labels["third_type"][third_type]
        return None

    def validate_and_link(self, query, model_output):
        result = {
            "primary_intent": "wash",
            "type": None,
            "second_type": None,
            "third_type": "none",
            "command_label": None
        }

        # Case: clarification
        if self.ask_que and self.cur_cloth_type:
            # BACK logic
            if query in BACK:
                print("User requested to go back or cancel. Resetting state.")
                msg = "リクエストをキャンセルしました。最初からやり直してください。"
                wav_path = self.tts_agent.synthesize(msg)
                self.stt_agent.audio_manager.play(wav_path)
                self.ask_que, self.cur_cloth_type, self.second_type = False, None, None
                return None
            result["type"] = self.cur_cloth_type
            cloth = self.cur_cloth_type
            if self.second_type:
                result["second_type"] = self.second_type
            else:
                result["second_type"] = None
            # Match clarification
            if cloth == "everyday":
              if self.second_type == "preferred":
                if query in BACK:
                  print("User requested to go back or cancel. Resetting state.")
                  msg = "リクエストをキャンセルしました。最初からやり直してください。"
                  wav_path = self.tts_agent.synthesize(msg)
                  self.stt_agent.audio_manager.play(wav_path)
                  self.ask_que, self.cur_cloth_type, self.second_type = False, None, None
                  return None
                elif query in QUIET: result["third_type"] = "quiet_at_night"
                elif query in ECO: result["third_type"] = "save_electricity"
                elif query in POWERFULL: result["third_type"] = "wash_thoroughly"
                else:
                  print("Sorry, I didn’t understand your choice. Let’s try again with a new request.")
                  msg = "申し訳ございませんが、ご選択いただいた内容が理解できませんでした。新しいリクエストでもう一度お試しください。"
                  wav_path = self.tts_agent.synthesize(msg)
                  self.stt_agent.audio_manager.play(wav_path)
                  print("申し訳ございませんが、ご選択いただいた内容が理解できませんでした。新しいリクエストでもう一度お試しください。")
                  self.ask_que, self.cur_cloth_type, self.second_type = False, None, None
                  return None
              else:
                if query in SEBUM: result["second_type"] = "sebum_sweat_yellowing"
                elif query in MUD_ODOR: result["second_type"] = "mud_food_spills_strong_odor"
                elif query in YELLOW: result["second_type"] = "yellowing"

            elif cloth == "delicates":
                if query in GENTLE_WASH: result["second_type"] = "gentle_wash"
                elif query in SEBUM: result["second_type"] = "target_sebum_sweat"
                elif query in YELLOW: result["second_type"] = "remove_yellowing"
                elif query in REFRESH: result["second_type"] = "light_refresh_overall"

            elif cloth == "towels":
                if query in FLUFFY_SOFT: result["second_type"] = "fluffy_soft"
                elif query in HURRY: result["second_type"] = "in_a_hurry"

            elif cloth == "sportswear":
                if query in YES or query in HURRY: result["second_type"] = "in_a_hurry"
                elif query in NO: result["second_type"] = "not_in_a_hurry"

            elif cloth == "bedding":
                if query == "blanket": result["second_type"] = "blanket"
                elif query in {"掛け布団", "羽毛布団"}: result["second_type"] = "warm_water_thorough"

            elif cloth == "whites":
                if query in SANITIZE_STRONG or query in YES: result["second_type"] = "sanitize"
                elif query in GENTLE_WASH or query in NO: result["second_type"] = "gentle"

            elif cloth == "care":
                if query in DEWRINKLE_DEODOR: result["second_type"] = "dewrinkle_deodorize"
                elif query in POLLEN: result["second_type"] = "pollen_sanitize_deodorize"
                elif query in RESTORE_REPELLENCY: result["second_type"] = "restore_water_repellency"

            elif cloth == "dry_only":
                if query in SOFT_DRY or query in YES: result["second_type"] = "soft"
                elif query in NO: result["second_type"] = "normal"

            # Invalid response
            if not result["second_type"]:
                print("Sorry, I didn’t understand your choice. Let’s try again with a new request.")
                msg = "申し訳ございませんが、ご選択いただいた内容が理解できませんでした。新しいリクエストでもう一度お試しください。"
                wav_path = self.tts_agent.synthesize(msg)
                self.stt_agent.audio_manager.play(wav_path)
                print("申し訳ございませんが、ご選択いただいた内容が理解できませんでした。新しいリクエストでもう一度お試しください。")
                self.ask_que, self.cur_cloth_type, self.second_type = False, None, None
                return None #result

            result["command_label"] = self.get_command_label(result["type"], result["second_type"], result["third_type"])
            if result["command_label"] == "none":
                print("Not able to select wash cycle, try again with proper statement.")
                msg = "洗濯コースを選択できません、正しい指示で再試行してください。"
                wav_path = self.tts_agent.synthesize(msg)
                self.stt_agent.audio_manager.play(wav_path)
            elif result["command_label"]:
                print(f" I will wash your {result['type']}, I am starting {result['command_label']} wash cycle.")
                msg = f"あなたの {result['type']}, を洗います。{result['command_label']} 洗浄サイクルを開始します。"
                wav_path = self.tts_agent.synthesize(msg)
                self.stt_agent.audio_manager.play(wav_path)
                print(msg)
            else:
                print("None of the option is matching, try again with proper statement.")
                msg = "どのオプションも一致しません。適切なステートメントでもう一度試してください。"
                wav_path = self.tts_agent.synthesize(msg)
                self.stt_agent.audio_manager.play(wav_path)
                print(msg)

            #self.ask_que, self.cur_cloth_type = False, None
            self.ask_que, self.cur_cloth_type, self.second_type = False, None, None
            return result

        # Case: first query from DistilBERT
        else:
          if model_output is not None:
            result.update(model_output)
            intent, wash_type, second_type, third_type = result["primary_intent"], result["type"], result["second_type"], result["third_type"]
            intent_conf = result.get("intent_confidence")
            types_conf = result.get("types_confidence")
            second_types_conf = result.get("second_types_confidence")
            third_types_conf = result.get("third_types_confidence")
            # You can now use these confidence scores as needed, e.g. print or log them
            # print(f"Intent confidence: {intent_conf}, Type confidence: {types_conf}, Second type confidence: {second_types_conf}, Third type confidence: {third_types_conf}")

            if intent == "general_info" or intent_conf < 0.5:
                result = self.text_agent.answer_query(query)[0]
                answer, similarity, emb_infer_time = result["answer"], result["similarity"], result["infer_time"]
                if similarity < SIMILARITY_THRESHOLD:
                    print(f"Similarity {similarity:.6f} below threshold, using Claude as fallback...")

                    claude_response = chat_with_claude(query)
                    fallback_msg = claude_response

                    print(f"Agent (Claude): {fallback_msg}")

                    tts_start = time.time()

                    # Use synthesize_stream_async for Claude fallback
                    async def play_stream():
                        await self.tts_agent.synthesize_stream_async(fallback_msg, audio_manager=self.stt_agent.audio_manager)

                    asyncio.run(play_stream())
                    tts_synthesis_time = time.time() - tts_start

                    # Play time is included in synthesize_stream_async stats
                else:
                    print(f"Agent Local: {answer}" + f" ({emb_infer_time * 1000:.3f} ms, Similarity: {similarity:.6f})")

                    tts_start = time.time()
                    wav_path = self.tts_agent.synthesize(answer)
                    tts_synthesis_time = time.time() - tts_start

                    # Play audio and track time
                    play_start = time.time()
                    self.stt_agent.audio_manager.play(wav_path)
                    play_time = time.time() - play_start

                    print(
                        f"TTS stats: synthesis={tts_synthesis_time * 1000:.3f}ms, playback={play_time * 1000:.3f}ms, total={(tts_synthesis_time + play_time) * 1000:.3f}ms")
                return None
            elif intent == "maintenance" and types_conf > 0.6:
                main_type = result["type"]
                if main_type in maintenance_json:
                    print(f"I am starting {maintenance_json[main_type]} maintenance cycle")
                    msg = f"{maintenance_json[main_type]} メンテナンスサイクルを開始します。"
                    wav_path = self.tts_agent.synthesize(msg)
                    self.stt_agent.audio_manager.play(wav_path)
                    # print(f"{maintenance_json[main_type]} メンテナンスサイクルを開始します。")
                else:
                    print("Maintenance type is unknown")
                    msg = "メンテナンスの種類が不明です"
                    wav_path = self.tts_agent.synthesize(msg)
                    self.stt_agent.audio_manager.play(wav_path)
                    # print("メンテナンスの種類が不明です")
                return None
            elif intent == "wash" and types_conf > 0.5:
                if wash_type in WASH_CYCLES:
                    result["command_label"] = self.get_command_label(wash_type, second_type, third_type)
                    if result["command_label"] == "none":
                        print("Need to ask for question")# Need more info:", question_ask.get(wash_type, "Please clarify."))
                        #self.ask_que, self.cur_cloth_type = True, wash_type
                    elif result["command_label"]:
                        print(f"I will wash your {wash_type}, I am starting {result['command_label']} wash cycle.")
                        msg = f"あなたの {wash_type} を洗います。{result['command_label']} 洗浄サイクルを開始します。"
                        wav_path = self.tts_agent.synthesize(msg)
                        self.stt_agent.audio_manager.play(wav_path)
                        # print(f"私があなたのものを洗います {wash_type}, 始めています {result['command_label']} 洗濯サイクル")
                    else:
                        print("Please try again with proper statement.")
                        msg = "正しい文言で再度お試しください。"
                        wav_path = self.tts_agent.synthesize(msg)
                        self.stt_agent.audio_manager.play(wav_path)
                        print("正しい文言で再度お試しください。")
                        self.ask_que, self.cur_cloth_type, self.second_type = False, None, None
                return result
            else:
                print("--INFO--: Intent is unknown:",intent)
                msg = "質問を繰り返してください。"
                wav_path = self.tts_agent.synthesize(msg)
                self.stt_agent.audio_manager.play(wav_path)
                # print("意図は不明")
                return None

    def infer_washbert(self, transcribed_text):
        # Mock model
        result = predict_onnx(transcribed_text)
        return result

    def process_query(self, query):
        if self.ask_que:
            #print("2nd query")
            updated_query = self.preprocess_japanese_text(query)
            return self.validate_and_link(updated_query, None)
        else:
            #print("1st query")
            model_output = self.infer_washbert(query)
            return self.validate_and_link(query, model_output)

"""
# Example usage
assistant = WashAssistantFSM()
first = assistant.process_query("Wash towels")   # asks clarification
second = assistant.process_query("fluffy")       # resolves cycle
third = assistant.process_query("xyz")           # invalid → resets
next = assistant.process_query("nothing")
"""