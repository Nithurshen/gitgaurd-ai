"""
Defines the main orchestration graph for GitGuard AI.

This module connects the Reviewer Agent and the Posting Tool into a
stateful LangGraph workflow. It implements the Human-in-the-Loop (HITL)
logic using checkpointers and interrupt_before breakpoints.
"""

import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from state import ReviewState
from agent import reviewer_node
from tools import post_pr_review


def poster_node(state: ReviewState) -> dict:
    """
    Node responsible for posting approved comments to GitHub.

    This node checks the 'review_approved' flag. If True, it converts
    the Pydantic comment models to dictionaries and invokes the GitHub tool.
    """
    if not state.review_approved:
        return {
            "messages": [("ai", "❌ Review NOT approved by human. No comments posted.")]
        }

    if not state.proposed_comments:
        return {"messages": [("ai", "✅ No issues found. Skipping comment posting.")]}

    # Convert Pydantic models to list of dicts for the tool
    comments_payload = [c.model_dump() for c in state.proposed_comments]

    # Execute the tool
    result = post_pr_review.invoke(
        {
            "repo_name": state.repo_name,
            "pr_number": state.pr_number,
            "comments": comments_payload,
        }
    )

    return {"messages": [("ai", f"🚀 {result}")]}


# --- Graph Construction ---
builder = StateGraph(ReviewState)

# Add Nodes
builder.add_node("reviewer", reviewer_node)
builder.add_node("poster", poster_node)

# Add Edges
builder.add_edge(START, "reviewer")
builder.add_edge("reviewer", "poster")
builder.add_edge("poster", END)

# --- Persistence & Compilation ---
# We use SQLite to persist the state between the "interrupt" and the "resume" actions
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)

# COMPILE WITH HITL:
# We interrupt *before* the 'poster' node runs. This gives the human
# a chance to inspect 'state.proposed_comments' and set 'review_approved=True'.
graph = builder.compile(checkpointer=memory, interrupt_before=["poster"])
