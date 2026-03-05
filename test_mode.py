###############################################################
# This separate script is created for test purpose only.
# It accepts text based input and produce text output only
###############################################################

import argparse
from core.embeddings import TextEmbeddingsAgent
from core.wash_fsm.text_washer_assistant import TextBasedWashAssistantFSM
# from core.text_to_speech import TextToSpeechAgent
from config import DEFAULT_QA_FILE, TEST_CSV_FILE

def load_queries_from_csv(file_path: str) -> list:
    import csv
    queries = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as f:
        csvreader = csv.reader(f)
        next(csvreader, None)  # Skip header

        for row in csvreader:
            if len(row) >= 3:
                queries.append(row[2])
                
    return queries


def main():
    parser = argparse.ArgumentParser(description="Test Mode: Process text-based queries")
    parser.add_argument(
        "--qa-file",
        type=str,
        default=DEFAULT_QA_FILE,
        help="Path to the QA file"
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU-only execution"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of CPU threads to use"
    )

    args = parser.parse_args()

    questions = load_queries_from_csv(TEST_CSV_FILE)
    
    intents = ["wash", "maintenance", "general_info"]
    # Initialize agents
    text_agent = TextEmbeddingsAgent(args.qa_file, cpu_only=args.cpu_only, cpu_cores=args.threads)
    # tts_agent = TextToSpeechAgent(tts_model=None, tts_voice=None)  # TTS is not used for output

    # Initialize FSM
    washer_agent = TextBasedWashAssistantFSM(text_agent)  # STT agent is bypassed
    
    for question in questions:
        print(f"[+] Processing query: {question}")
        # Process the query through the FSM and get the response
        try:
            response = washer_agent.process_query(question)
            if response is not None:
                print("Getting Response...")
                intent = response.get("predicted_intent")
                type = response.get("predicted_type")
                intent_conf = response.get("intent_confidence")
                type_conf = response.get("type_confidence")
                print(f"Intent: {intent} (Intent conf: {intent_conf:.3f}), Type : {type} (Type Conf: {type_conf:.3f})")
                
            elif response.get("predicted_second_type") is not None:
                second_type = response.get("predicted_second_type")
                second_type_conf = response.get("second_type_confidence")
                print(f"Second Type: {second_type} (Second Type Conf: {second_type_conf:.3f})")
            else:
                print("No valid response generated for this query.")
        
        except Exception as e:
            print(f"Error processing query: {e}")
        print("-" * 50)  
        
        
if __name__ == "__main__":
    main()