"""
CLI for the unicorn startups chatbot. Run: python cli.py
Uses in-memory checkpointer; a new thread_id is generated for each run (each time you start the CLI).
Logs each turn and flushes Langfuse on exit.
"""
import json
import sys
import uuid

from src.chatbot import conversation_turn, flush_langfuse
from src.rag import get_embedding_model
from src.metrics import get_metrics


def main():
    print("Unicorn Startups Chatbot (dataset: data/tracxn.csv)")
    print("Loading embedding model (uses cache if you ran ingest.py before)...")
    try:
        get_embedding_model()
    except Exception as e:
        print(
            f"Could not load embedding model: {e}\n"
            "Run 'python ingest.py' once when online to cache the model, then try again."
        )
        return
    print("Ask about companies, sectors, locations. Type 'quit' or 'exit' to stop.\n")

    thread_id = str(uuid.uuid4())
    history = []

    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        try:
            reply, history = conversation_turn(thread_id, user_input, history=history)
        except Exception as e:
            print(f"Error: {e}")
            reply, history = "An error occurred. Please try again.", history


        # print("--------------------------------")
        print("\033[92m", end="")
        print(f"Bot: {reply}\n")
        print("\033[0m", end="")
        # print("--------------------------------")
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # print("\033[94m", end="")
        # print("history", json.dumps(history, indent=4))
        # print("\033[0m", end="")
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    # Flush Langfuse so traces are sent (short-lived app)
    flush_langfuse()
    m = get_metrics()
    if m.total_queries > 0:
        print("\n--- Chat metrics (this run) ---")
        print(m.summary_for_display())
        print(json.dumps(m.summary(), indent=2))
    else:
        print("\n--- Chat metrics ---")
        print("No queries in this session.")


if __name__ == "__main__":
    main()
    sys.exit(0)
