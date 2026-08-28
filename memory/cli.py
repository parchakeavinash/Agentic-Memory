import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from memory.agent import MemoryAgent
from memory.config import settings


def print_banner(current_user: str, current_session: str):
    print("=" * 65)
    print("   🧠 AGENT MEMORY SYSTEM (SHORT + SUMMARY + EPISODIC + SEMANTIC) 🧠")
    print("=" * 65)
    print(f" • User ID            : {current_user} (Multi-Tenant Isolation)")
    print(f" • Session ID         : {current_session}")
    print(f" • LLM Model          : {settings.groq_model}")
    print(f" • Embedding Model    : {settings.gemini_embedding_model}")
    print(f" • Memory Window (K)  : {settings.memory_window} messages")
    print(f" • Top-K Episodes     : {settings.episodic_top_k} (Min Similarity: {settings.episodic_min_similarity})")
    print("=" * 65)
    print("Commands:")
    print("  /history            - View database messages & sliding window status")
    print("  /summary            - View condensed summary of older evicted turns")
    print("  /facts              - List all semantic facts (persistent user knowledge)")
    print("  /forget <key>       - Delete a specific semantic fact by key")
    print("  /episodes           - List all extracted episodic memories in vector DB")
    print("  /search <query>     - Semantic vector search for past episodes")
    print("  /create-episode     - Distill current session into a new structured episode")
    print("  /prompt             - Inspect the exact prompt & memory injection payload")
    print("  /stats              - View memory system metrics")
    print("  /user <id>          - Switch to another user ID (data isolation test)")
    print("  /session <id>       - Switch to another session ID")
    print("  /window <size>      - Change sliding memory window size")
    print("  /clear              - Clear short-term memory & summary for current session")
    print("  /help               - Show this command menu")
    print("  /exit               - Quit the application")
    print("=" * 65)
    print("Tip: Just chat naturally. Semantic facts are auto-extracted from your messages.")
    print("Use /facts to see what the agent has learned about you.\n")


def display_facts(agent: MemoryAgent, current_user: str):
    facts = agent.semantic.list_facts(user_id=current_user)

    print("\n" + "=" * 65)
    print(f"🧩 SEMANTIC MEMORY (Persistent Facts) — User: '{current_user}' — Total: {len(facts)}")
    print("=" * 65)

    if not facts:
        print(" [No semantic facts stored yet. Just chat! Facts are auto-extracted from your messages.]")
        print("=" * 65 + "\n")
        return

    for fact in facts:
        updated = fact.updated_at.strftime("%Y-%m-%d %H:%M") if fact.updated_at else "N/A"
        print(f"  {fact.key:<35} = {fact.value}")
        if fact.source:
            src_preview = fact.source[:65] + ("..." if len(fact.source) > 65 else "")
            print(f"  {'':35}   (from: \"{src_preview}\")")
        print(f"  {'':35}   [updated: {updated}]")
        print()
    print("=" * 65 + "\n")


def display_history(agent: MemoryAgent, current_user: str, current_session: str):
    messages = agent.get_history(session_id=current_session, user_id=current_user)
    stats = agent.get_stats(session_id=current_session, user_id=current_user)
    window_size = agent.memory.default_window_size

    print("\n" + "-" * 55)
    print(f"📜 CONVERSATION HISTORY (User: '{current_user}', Session: '{current_session}')")
    print(f"Total Stored: {stats['total_stored_messages']} | Window Size: {window_size} | Active in Context: {stats['active_in_prompt']}")
    print("-" * 55)

    if not messages:
        print(" [No messages stored in memory yet for this user/session]")
        print("-" * 55 + "\n")
        return

    cutoff_index = max(0, len(messages) - window_size)

    for i, msg in enumerate(messages):
        is_in_window = i >= cutoff_index
        tag = "[ACTIVE IN PROMPT]" if is_in_window else "[CONDENSED INTO SUMMARY]"
        speaker = "👤 User" if msg.role == "user" else "🤖 Agent" if msg.role == "assistant" else "⚙️ System"
        print(f"{i+1}. {speaker} {tag}:")
        print(f"   {msg.content}")
        print()
    print("-" * 55 + "\n")


def display_summary(agent: MemoryAgent, current_user: str, current_session: str):
    summary = agent.get_summary(session_id=current_session, user_id=current_user)
    stats = agent.get_stats(session_id=current_session, user_id=current_user)

    print("\n" + "=" * 60)
    print(f"📑 CONVERSATION SUMMARY (User: '{current_user}', Session: '{current_session}')")
    print(f"Total Messages: {stats['total_stored_messages']} | Evicted from Window: {stats['evicted_from_window']}")
    print("=" * 60)

    if summary:
        print(summary)
    else:
        print("[No summary generated yet. Summary activates once message count exceeds the sliding window size.]")
    print("=" * 60 + "\n")


def display_episodes(agent: MemoryAgent, current_user: str, current_session: str):
    episodes = agent.list_episodes(user_id=current_user)

    print("\n" + "=" * 65)
    print(f"🏛️ STORED EPISODIC MEMORIES (User: '{current_user}') — Total: {len(episodes)}")
    print("=" * 65)

    if not episodes:
        print(" [No episodic memories found for this user. Type /create-episode to distill!]")
        print("=" * 65 + "\n")
        return

    for i, ep in enumerate(episodes):
        date_str = ep.timestamp.strftime('%Y-%m-%d %H:%M') if ep.timestamp else 'N/A'
        print(f"Episode #{i+1} | ID: {ep.episode_id} | Session: '{ep.session_id}' | Date: {date_str}")
        if ep.start_message_id and ep.end_message_id:
            print(f"🔗 Provenance: Messages #{ep.start_message_id} to #{ep.end_message_id}")
        print(f"📖 Summary: {ep.summary}")
        if ep.events:
            print("⚡ Events:")
            for ev in ep.events:
                print(f"   • {ev}")
        if ep.topics:
            print(f"🏷️ Topics : {', '.join(ep.topics)}")
        print("-" * 65)
    print("=" * 65 + "\n")


def search_episodes_interactive(agent: MemoryAgent, current_user: str, query: str):
    if not query:
        print("Usage: /search <query text>")
        return

    print(f"\n🔍 Searching episodic vector memory for User '{current_user}': '{query}'...")
    results = agent.search_episodes(query=query, user_id=current_user, top_k=3)

    print("\n" + "=" * 65)
    print(f"🎯 EPISODIC VECTOR SEARCH RESULTS ({len(results)} matches)")
    print("=" * 65)

    if not results:
        print(" [No relevant episodes found matching query above similarity threshold]")
        print("=" * 65 + "\n")
        return

    for i, res in enumerate(results):
        print(f"Match #{i+1} | Similarity: {res['similarity']:.4f} | Date: {res['timestamp'][:10] if res.get('timestamp') else 'N/A'}")
        if res.get("start_message_id") and res.get("end_message_id"):
            print(f"🔗 Provenance: Messages #{res['start_message_id']} to #{res['end_message_id']}")
        print(f"📖 Summary: {res['summary']}")
        if res.get("events"):
            print("⚡ Key Events:")
            for ev in res["events"][:3]:
                print(f"   • {ev}")
        if res.get("topics"):
            print(f"🏷️ Topics  : {', '.join(res['topics'])}")
        print("-" * 65)
    print("=" * 65 + "\n")


def create_episode_interactive(agent: MemoryAgent, current_user: str, current_session: str):
    print(f"\n⏳ Distilling conversation from User '{current_user}', Session '{current_session}' into structured episode...")
    episode_data = agent.create_episode(session_id=current_session, user_id=current_user)
    if not episode_data:
        print("❌ Could not extract episode. Make sure there are messages in the active session.\n")
        return

    print("\n" + "=" * 65)
    print("✨ EPISODIC MEMORY CREATED & EMBEDDED SUCCESSFULLY!")
    print("=" * 65)
    print(f" • Episode ID  : {episode_data['episode_id']}")
    print(f" • User ID     : {episode_data['user_id']}")
    print(f" • Session ID  : {episode_data['session_id']}")
    print(f" • Timestamp   : {episode_data['timestamp']}")
    if episode_data.get("start_message_id") and episode_data.get("end_message_id"):
        print(f" • Provenance  : Messages #{episode_data['start_message_id']} to #{episode_data['end_message_id']}")
    print(f" • Summary     : {episode_data['summary']}")
    print(" • Events      :")
    for ev in episode_data.get("events", []):
        print(f"   - {ev}")
    print(f" • Topics      : {', '.join(episode_data.get('topics', []))}")
    print("=" * 65 + "\n")


def display_last_prompt(agent: MemoryAgent, current_user: str, current_session: str):
    debug_info = agent.get_last_prompt_debug()
    if not debug_info:
        print("\n[No prompts have been sent to the LLM yet in this runtime session.]")
        print("Send a message first, then use /prompt to inspect the exact payload!\n")
        return

    print("\n" + "=" * 65)
    print("       🔍 LLM PROMPT PAYLOAD (MEMORY INJECTION DEBUG) 🔍")
    print("=" * 65)
    print(f" • User ID               : {debug_info.get('user_id', current_user)}")
    print(f" • Session ID            : {debug_info['session_id']}")
    print(f" • Total Stored in DB    : {debug_info['total_stored_messages']} messages")
    print(f" • Sliding Window Limit  : {debug_info['window_size']} messages")
    print(f" • Summary Injected      : {'YES' if debug_info.get('has_summary') else 'NO'}")
    print(f" • Episodic Injected     : {debug_info.get('episodic_count', 0)} episodes")
    print(f" • History Injected      : {debug_info['history_injected_count']} messages")
    print(f" • Total Injected Items  : {debug_info['total_prompt_items']} items")
    print("-" * 65)
    print(" [Exact Sequence of Messages Sent to the Model]:\n")

    for i, msg in enumerate(debug_info['messages']):
        role_type = msg.__class__.__name__
        content_preview = msg.content
        if role_type == "SystemMessage":
            if "Summary of earlier conversation" in content_preview:
                speaker = f"📑 [{i}] System (Condensed Conversation Summary)"
            elif "Relevant Past Episodes" in content_preview:
                speaker = f"🏛️ [{i}] System (Episodic Vector Retrieval)"
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
    current_user = settings.user_id
    current_session = settings.session_id
    agent = MemoryAgent(user_id=current_user)

    print_banner(current_user, current_session)

    while True:
        try:
            user_input = input(f"[{current_user}:{current_session}] User > ").strip()
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
                print_banner(current_user, current_session)
            elif cmd in ("/history", "/mem"):
                display_history(agent, current_user, current_session)
            elif cmd == "/summary":
                display_summary(agent, current_user, current_session)
            elif cmd == "/facts":
                display_facts(agent, current_user)
            elif cmd == "/forget":
                if not arg:
                    print("Usage: /forget <key>  (e.g. /forget favorite_language)")
                else:
                    deleted = agent.semantic.delete_fact(user_id=current_user, key=arg)
                    if deleted:
                        print(f"\n🗑️ Deleted semantic fact: '{arg}' for User '{current_user}'\n")
                    else:
                        print(f"\n[No fact found with key '{arg}' for User '{current_user}']\n")
            elif cmd == "/episodes":
                display_episodes(agent, current_user, current_session)
            elif cmd in ("/search", "/search-episodes", "/find"):
                search_episodes_interactive(agent, current_user, arg)
            elif cmd in ("/create-episode", "/save-episode"):
                create_episode_interactive(agent, current_user, current_session)
            elif cmd == "/prompt":
                display_last_prompt(agent, current_user, current_session)
            elif cmd == "/stats":
                stats = agent.get_stats(session_id=current_session, user_id=current_user)
                print(f"\n📊 Memory Stats: {stats}\n")
            elif cmd == "/clear":
                count = agent.clear_memory(session_id=current_session, user_id=current_user)
                print(f"\n🧹 Memory cleared! Removed {count} messages and summary for User '{current_user}', Session '{current_session}'.\n")
            elif cmd == "/user":
                if not arg:
                    print("Usage: /user <user_id>")
                else:
                    current_user = arg
                    agent.user_id = current_user
                    print(f"\n👤 Switched to User: '{current_user}'\n")
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
            response = agent.chat(user_input=user_input, session_id=current_session, user_id=current_user)
            print(f"\n🤖 Agent: {response}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
