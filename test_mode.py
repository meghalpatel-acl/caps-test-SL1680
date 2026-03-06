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
    
    # Load all columns from the CSV
    import csv
    test_rows = []
    with open(TEST_CSV_FILE, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_rows.append(row)
    
    print(f"=== Starting Automated Batch Test ({len(test_rows)}) queries ===\n")
    
    for row in test_rows:
        japanese_query = row['Question']
        target_option = row['second_type']
        
        print(f"[TESTING] Initial Query: {japanese_query}")
        
        try:
            # 1. process initial query (sets intent and type)
            washer_agent.process_query(japanese_query)
            
            # 2. Automatically select second type (Based on CSV 'second_type' column)
            if washer_agent.state == "wait_second_input":
                print(f"    -> Auto-selecting option: {target_option}")
                washer_agent.process_query(target_option)
                
            # 3. Automatically confirm (Finalize the cycle)
            if washer_agent.state == "wait_confirm_second":
                print(f"    -> Auto-confirming: はい")
                washer_agent.process_query("はい")
        except Exception as e:
            print (f"[!] Error: {e}")

        # Ensure FSM is reset for next CSV row
        # washer_agent.reset()
        print("-" * 60 + "\n")
        
    print(f"\n=== Batch Test Complete ===")
        
    # for question in questions:
    #     print(f"[+] Processing query: {question}")
    #     # Process the query through the FSM and get the response
    #     try:
    #         response = washer_agent.process_query(question)
    #         if response is not None:
    #             print("Getting Response...")
    #             intent = response.get("predicted_intent")
    #             type = response.get("predicted_type")
    #             intent_conf = response.get("intent_confidence")
    #             type_conf = response.get("type_confidence")
    #             print(f"Intent: {intent} (Intent conf: {intent_conf:.3f}), Type : {type} (Type Conf: {type_conf:.3f})")
                
    #         elif response.get("predicted_second_type") is not None:
    #             second_type = response.get("predicted_second_type")
    #             second_type_conf = response.get("second_type_confidence")
    #             print(f"Second Type: {second_type} (Second Type Conf: {second_type_conf:.3f})")
    #         else:
    #             print("No valid response generated for this query.")
        
    #     except Exception as e:
    #         print(f"Error processing query: {e}")
    #     print("-" * 50)  
        
        
if __name__ == "__main__":
    main()