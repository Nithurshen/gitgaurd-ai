"""
Defines the core Reviewer Agent logic for GitGuard AI.

This module implements the LangGraph node responsible for fetching PR diffs
and generating structured code review comments using an LLM.
"""

from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.state import ReviewState, PullRequestComment
from src.tools import fetch_pr_diff

# Initialize the model with low temperature for deterministic analysis
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class ReviewResponse(BaseModel):
    """Internal wrapper schema to ensure the LLM returns a list of comments."""

    comments: List[PullRequestComment] = Field(
        default_factory=list, description="List of identified issues."
    )


def reviewer_node(state: ReviewState) -> dict:
    """
    Main agent node: Fetches diff (if needed) -> Analyzes code -> Outputs comments.

    Returns:
        dict: Updates for the 'pr_diff' and 'proposed_comments' state keys.
    """
    # 1. Ensure we have the diff
    current_diff = state.pr_diff
    if not current_diff:
        # Invoke the tool directly to populate state
        current_diff = fetch_pr_diff.invoke(
            {"repo_name": state.repo_name, "pr_number": state.pr_number}
        )

    # 2. Configure the analysis prompt
    system_prompt = """You are a strict Senior Code Reviewer.
    Analyze the git diff provided below for:
    1. Security Vulnerabilities (SQLi, XSS, Secrets) - Severity: Critical
    2. Logic Bugs & Race Conditions - Severity: Major
    3. Performance Bottlenecks - Severity: Major
    4. Code Style & Best Practices (PEP8, DRY) - Severity: Minor/Nitpick

    Output a structured list of comments. Only comment on changed lines.
    If the code looks good, return an empty list.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "Repository: {repo}\nDiff:\n{diff}"),
        ]
    )

    # 3. Create structured chain
    # We wrap the output in ReviewResponse to handle the list correctly
    structured_llm = llm.with_structured_output(ReviewResponse)
    chain = prompt | structured_llm

    # 4. Execute analysis
    result = chain.invoke({"repo": state.repo_name, "diff": current_diff})

    # 5. Return state updates
    return {
        "pr_diff": current_diff,
        "proposed_comments": result.comments,
    }
