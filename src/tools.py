"""
Implements the GitHub interaction tools for GitGuard AI.

This module uses PyGithub to fetch raw diffs from Pull Requests and
post structured review comments back to the repository.
"""

import os
import requests
from typing import List, Dict, Any
from github import Github, GithubException, Auth
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# Global GitHub client instance
# Updated to use Auth.Token to resolve DeprecationWarning
GITHUB_CLIENT = Github(auth=Auth.Token(os.getenv("GITHUB_TOKEN")))


@tool
def fetch_pr_diff(repo_name: str, pr_number: int) -> str:
    """
    Fetches the raw text of the git diff for a specific Pull Request.
    """
    try:
        repo = GITHUB_CLIENT.get_repo(repo_name)
        pull_request = repo.get_pull(pr_number)

        # PyGithub returns a URL for the diff; we must fetch the raw content
        response = requests.get(pull_request.diff_url)
        response.raise_for_status()

        return response.text
    except (GithubException, requests.RequestException) as e:
        return f"Error fetching PR diff: {str(e)}"


@tool
def post_pr_review(repo_name: str, pr_number: int, comments: List[Dict[str, Any]]) -> str:
    """
    Posts a batch of review comments to a GitHub Pull Request.

    Args:
        repo_name: The "owner/repo" string.
        pr_number: The integer ID of the Pull Request.
        comments: List of dicts matching the PullRequestComment schema.
    """
    try:
        repo = GITHUB_CLIENT.get_repo(repo_name)
        pull_request = repo.get_pull(pr_number)

        # PyGithub requires the commit object to anchor comments
        latest_commit = pull_request.get_commits().reversed[0]

        formatted_comments = []
        for c in comments:
            formatted_comments.append({
                "path": c.get("file_path"),
                # "position" is legacy and conflicts with "line" in the new API
                # We strictly use "line" and "side"='RIGHT' (new code)
                "body": f"[{c.get('severity')}] {c.get('body')}",
                "line": c.get("line_number"),
                "side": "RIGHT"
            })

        pull_request.create_review(
            commit=latest_commit,
            body="GitGuard AI Review Summary: Issues detected.",
            event="COMMENT",
            comments=formatted_comments
        )
        return "Review submitted successfully."

    except GithubException as e:
        return f"Error posting review: {str(e)}"