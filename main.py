"""
Main entry point for running GitGuard AI.

This script demonstrates the full application lifecycle:
1. Starts the graph with a PR.
2. Reviewer Agent fetches diff and proposes comments.
3. Execution pauses (HITL) for human verification.
4. User reviews comments in the console and approves/rejects.
5. Graph resumes to post comments to GitHub (if approved).
"""

import uuid
import os
from src.graph import graph

def run_gitguard(repo_name: str, pr_number: int):
    """
    Runs the GitGuard AI workflow for a specific Pull Request.
    """
    # Create a unique thread ID for this specific review session
    # This is required by the SqliteSaver checkpointer
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(f"🚀 Starting GitGuard AI for PR #{pr_number} in {repo_name}...")
    print(f"   Thread ID: {thread_id}")

    # --- Phase 1: Analysis ---
    initial_state = {
        "repo_name": repo_name,
        "pr_number": pr_number
    }

    print("\n--- Phase 1: Fetching & Analyzing ---")
    # Stream events until the interruption point
    for event in graph.stream(initial_state, config):
        for node, values in event.items():
            print(f"🤖 Node '{node}' executed.")

    # --- Phase 2: Human Review (HITL) ---
    # We fetch the current state to see if we are paused
    snapshot = graph.get_state(config)

    # snapshot.next will be non-empty if the graph is paused at an interrupt
    if snapshot.next:
        print("\n⏸️  Workflow Paused for Human Review")

        current_state = snapshot.values
        comments = current_state.get("proposed_comments", [])

        if not comments:
            print("✅ Agent found no issues. Process complete.")
            return

        print(f"\n📝 Proposed {len(comments)} Comments:")
        for i, c in enumerate(comments, 1):
            print(f"{i}. [{c.severity}] {c.file_path}:{c.line_number}")
            print(f"   \"{c.body}\"")
            print("-" * 40)

        # Simple CLI interaction for the "Human" part of the loop
        user_input = input("\nApprove posting these comments? (yes/no): ").strip().lower()

        if user_input in ["yes", "y"]:
            print("\n--- Phase 3: Posting to GitHub ---")

            # CRITICAL HITL STEP: Update the state to signal approval
            # This update is applied to the *current* step, effectively modifying
            # the state before the 'poster' node runs.
            graph.update_state(config, {"review_approved": True})

            # Resume execution by passing None (instructions to proceed)
            for event in graph.stream(None, config):
                 for node, values in event.items():
                    print(f"🤖 Node '{node}' executed.")
                    if "messages" in values:
                         last_msg = values["messages"][-1]
                         # Handle both Message objects (with .content) and tuples (role, content)
                         if hasattr(last_msg, "content"):
                             print(f"   Message: {last_msg.content}")
                         elif isinstance(last_msg, (tuple, list)) and len(last_msg) > 1:
                             print(f"   Message: {last_msg[1]}")
                         else:
                             print(f"   Message: {str(last_msg)}")
        else:
            print("\n❌ Review rejected by user. No comments posted.")

if __name__ == "__main__":
    # Allow configuration via environment variables or interactive input
    TARGET_REPO = os.getenv("TARGET_REPO")
    try:
        TARGET_PR = int(os.getenv("TARGET_PR", "0"))
    except ValueError:
        TARGET_PR = 0

    if not TARGET_REPO or TARGET_PR == 0:
        print("⚠️  Configuration needed.")
        TARGET_REPO = input("Enter Repo (format: owner/name): ").strip()
        TARGET_PR = int(input("Enter PR Number: ").strip())

    if TARGET_REPO and TARGET_PR:
        run_gitguard(TARGET_REPO, TARGET_PR)