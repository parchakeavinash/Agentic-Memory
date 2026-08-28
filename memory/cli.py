import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from memory.agent import MemoryAgent
from memory.config import settings


def print_banner():
    print("=" * 60)
    print("   🧠 AGENT SHORT-TERM MEMORY DEMONSTRATION & CLI 🧠")
    print("=" * 60)
    print(f" • Model          : {settings.groq_model}")
    print(f" • Session ID     : {settings.session_id}")
    print(f" • Memory Window  : {settings.memory_window} messages (Sliding Window Buffer)")
    print("=" * 60)
    print("Commands:")
    print("  /history       - View full database history & active sliding window")
    print("  /summary       - View condensed conversation summary of evicted turns")
    print("  /prompt        - Inspect the exact message payload sent to the LLM")
    print("  /stats         - View memory window metrics")
    print("  /session <id>  - Switch to another session ID")
    print("  /window <size> - Change sliding memory window size")
    print("  /clear         - Clear short-term memory for current session")
    print("  /help          - Show this command menu")
    print("  /exit          - Quit the application")
    print("=" * 60)
    print("Tip: Tell the agent a fact (e.g. 'My favorite color is emerald green'),")
    print("then ask questions across multiple turns to see how short-term memory works!\n")


def display_history(agent: MemoryAgent, current_session: str):
    messages = agent.get_history(session_id=current_session)
    stats = agent.get_stats(session_id=current_session)
    window_size = agent.memory.default_window_size

    print("\n" + "-" * 50)
    print(f"📜 MEMORY HISTORY FOR SESSION: '{current_session}'")
    print(f"Total Stored: {stats['total_stored_messages']} | Window Size: {window_size} | Active in Context: {stats['active_in_prompt']}")
    print("-" * 50)

    if not messages:
        print(" [No messages stored in memory yet]")
        print("-" * 50 + "\n")
        return

    # Determine cutoff index for sliding window
    cutoff_index = max(0, len(messages) - window_size)

    for i, msg in enumerate(messages):
        is_in_window = i >= cutoff_index
        tag = "[ACTIVE IN PROMPT]" if is_in_window else "[CONDENSED INTO SUMMARY]"
        speaker = "👤 User" if msg.role == "user" else "🤖 Agent" if msg.role == "assistant" else "⚙️ System"
        print(f"{i+1}. {speaker} {tag}:")
        print(f"   {msg.content}")
        print()
    print("-" * 50 + "\n")


def display_summary(agent: MemoryAgent, current_session: str):
    summary = agent.get_summary(session_id=current_session)
    stats = agent.get_stats(session_id=current_session)

    print("\n" + "=" * 60)
    print(f"📑 CONVERSATION SUMMARY FOR SESSION: '{current_session}'")
    print(f"Total Messages: {stats['total_stored_messages']} | Evicted from Window: {stats['evicted_from_window']}")
    print("=" * 60)

    if summary:
        print(summary)
    else:
        print("[No summary generated yet. Summary activates once message count exceeds the sliding window size.]")
    print("=" * 60 + "\n")


def display_last_prompt(agent: MemoryAgent, current_session: str):
    debug_info = agent.get_last_prompt_debug()
    if not debug_info:
        print("\n[No prompts have been sent to the LLM yet in this runtime session.]")
        print("Send a message first, then use /prompt to inspect the exact payload!\n")
        return

    print("\n" + "=" * 65)
    print("       🔍 LLM PROMPT PAYLOAD (MEMORY INJECTION DEBUG) 🔍")
    print("=" * 65)
    print(f" • Session ID            : {debug_info['session_id']}")
    print(f" • Total Stored in DB    : {debug_info['total_stored_messages']} messages")
    print(f" • Sliding Window Limit  : {debug_info['window_size']} messages")
    print(f" • Summary Injected      : {'YES' if debug_info.get('has_summary') else 'NO'}")
    print(f" • History Injected      : {debug_info['history_injected_count']} messages")
    print(f" • Total Injected Items  : {debug_info['total_prompt_items']} items (System + Summary + History + User Input)")
    print("-" * 65)
    print(" [Exact Sequence of Messages Sent to the Model]:\n")

    for i, msg in enumerate(debug_info['messages']):
        role_type = msg.__class__.__name__
        content_preview = msg.content
        if role_type == "SystemMessage":
            if "Summary of earlier conversation" in content_preview:
                speaker = f"📑 [{i}] System (Condensed Conversation Summary)"
            else:
                speaker = f"⚙️ [{i}] System Prompt"
        elif role_type == "AIMessage":
            speaker = f"🤖 [{i}] Assistant (History from Short-Term Memory)"
        elif role_type == "HumanMessage" and i == len(debug_info['messages']) - 1:
            speaker = f"👤 [{i}] User (Current Turn Input)"
        else:
            speaker = f"👤 [{i}] User (History from Short-Term Memory)"

        print(f"{speaker}:")
        print(f"    {content_preview}")
        print()
    print("=" * 65 + "\n")


def main():
    agent = MemoryAgent()
    current_session = settings.session_id

    print_banner()

    while True:
        try:
            user_input = input(f"[{current_session}] User > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            sys.exit(0)

        if not user_input:
            continue

        # Handle Commands
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("/exit", "/quit"):
                print("Exiting. Goodbye!")
                break
            elif cmd == "/help":
                print_banner()
            elif cmd in ("/history", "/mem"):
                display_history(agent, current_session)
            elif cmd == "/summary":
                display_summary(agent, current_session)
            elif cmd == "/prompt":
                display_last_prompt(agent, current_session)
            elif cmd == "/stats":
                stats = agent.get_stats(session_id=current_session)
                print(f"\n📊 Memory Stats: {stats}\n")
            elif cmd == "/clear":
                count = agent.clear_memory(session_id=current_session)
                print(f"\n🧹 Memory cleared! Removed {count} messages and summary for session '{current_session}'.\n")
            elif cmd == "/session":
                if not arg:
                    print("Usage: /session <session_id>")
                else:
                    current_session = arg
                    print(f"\n🔄 Switched to session: '{current_session}'\n")
            elif cmd == "/window":
                try:
                    new_size = int(arg)
                    agent.memory.default_window_size = new_size
                    print(f"\n⚙️ Memory window size updated to {new_size} messages.\n")
                except ValueError:
                    print("Usage: /window <integer>")
            else:
                print(f"Unknown command: {cmd}. Type /help for available commands.")
            continue

        # Process standard conversation turn
        try:
            response = agent.chat(user_input=user_input, session_id=current_session)
            print(f"\n🤖 Agent: {response}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
