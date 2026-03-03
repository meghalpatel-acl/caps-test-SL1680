import os
import anthropic
from dotenv import load_dotenv
import argparse
import time
from datetime import datetime, timedelta
from prompt_config import SYSTEM_PROMPT, STATIC_CONTEXT

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("No API key found. Please set ANTHROPIC_API_KEY in your .env file")

# Initialize the Anthropic client with your API key
client = anthropic.Anthropic(api_key=api_key)

# Disable HTTP request logging
import logging
# Set the logging level for httpx (used by Anthropic's client) to WARNING or higher to suppress INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)
# Additionally, set logging level for urllib3 which might also be used
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Global variables for caching
system_prompt_cache = {}
response_cache = {}

def parse_ttl(ttl_str):
    """Parse TTL string (e.g., '5m', '1h') to seconds."""
    return 600  # Default 10 minutes

def create_rag_prompt(query, cache_control=None):
    """Create a RAG prompt with query caching."""
    # Set default cache control if not provided
    if cache_control is None:
        cache_control = {
            "type": "ephemeral",
            "ttl": "10m"
        }
    
    # Create a cache key for this system prompt
    cache_key = "system_prompt"
    now = datetime.now()
    
    # Check if system prompt is already cached and not expired
    if cache_key in system_prompt_cache:
        cached_data = system_prompt_cache[cache_key]
        if now < cached_data["expiry"]:
            # print("Using cached system prompt...")
            return cached_data["content"], query
    
    print("Creating new system prompt...")
    
    # Use the system prompt directly from prompt_config.py
    system_prompt = SYSTEM_PROMPT
    
    # Cache the system prompt
    ttl_seconds = parse_ttl(cache_control["ttl"])
    expiry = now + timedelta(seconds=ttl_seconds)
    system_prompt_cache[cache_key] = {
        "content": system_prompt,
        "expiry": expiry,
        "type": cache_control["type"]
    }
    
    return system_prompt, query

def chat_with_claude(query, model="claude-3-haiku-20240307", cache_control=None):
    """Send a query to Claude."""
    # Set default cache control
    if cache_control is None:
        cache_control = {
            "type": "ephemeral",
            "ttl": "30m"
        }
    # Create a cache key from the query
    cache_key = f"{query}_{model}"
    now = datetime.now()
    # Check if response is already cached and not expired
    if cache_key in response_cache:
        cached_data = response_cache[cache_key]
        if now < cached_data["expiry"]:
            print("Using cached response...")
            return cached_data["content"]
    # Create RAG prompt with caching - system prompt is now cached
    system_prompt, user_query = create_rag_prompt(query, cache_control)
    print(f"Making API call to Claude ({model})...")
    messages = [{"role": "user", "content": user_query}]
    # Measure API call latency
    start_time = time.time()
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=messages
    )
    end_time = time.time()
    latency = end_time - start_time
    print(f"Claude API latency: {latency:.2f} seconds")
    response = message.content[0].text
    # Cache the response with expiration time
    ttl_seconds = parse_ttl(cache_control["ttl"])
    expiry = now + timedelta(seconds=ttl_seconds)
    response_cache[cache_key] = {
        "content": response,
        "expiry": expiry,
        "type": cache_control["type"]
    }
    return response

def interactive_chat(model="claude-3-haiku-20240307", cache_ttl="5m"):
    """Run an interactive chat session with Claude."""
    cache_control = {
        "type": "ephemeral",
        "ttl": cache_ttl
    }
    print(f"Using preconfigured prompt from prompt_config.py")
    print(f"Using model: {model}")
    print("Type 'exit' to end the conversation.")
    print("=" * 50)
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'exit':
            break
        print("\nClaude is thinking...")
        response = chat_with_claude(
            user_input, 
            model=model,
            cache_control=cache_control
        )
        print(f"\nClaude: {response}")
        print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description='Claude RAG Chatbot')
    parser.add_argument('--model', '-m', default='claude-3-haiku-20240307', help='Claude model to use')
    parser.add_argument('--ttl', '-t', default='5m', help='Cache TTL (e.g., 5m, 1h)')
    args = parser.parse_args()
    
    interactive_chat(args.model, args.ttl)

if __name__ == "__main__":
    main()
