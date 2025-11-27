# Overview of MAT496

In this course, I have primarily learned Langgraph. This is helpful tool to build apps which can process unstructured `text`, find information I am looking for, and present the format we choose. Some specific topics we have covered are:

- Prompting
- Structured Output 
- Semantic Search
- Retreaval Augmented Generation (RAG)
- Tool calling LLMs & MCP
- Langgraph: State, Nodes, Graph

I also learned that Langsmith is a nice tool for debugging Langgraph codes.

------

# Capstone Project objective

The first purpose of the capstone project is to give a chance to revise all the major above listed topics. The second purpose of the capstone is to show my creativity. Think about all the problems I can not have solved earlier, but are not possible to solve with the concepts learned in this course. For example, I can use LLM to analyse all kinds of news: sports news, financial news, political news. Another example, I can use LLMs to build a legal assistant. Pretty much anything which requires lots of reading, can be outsourced to LLMs.


-------------------------

# Project report Template

## Title: GitGuard AI
## Overview

GitGuard AI is an intelligent, multi-agent system designed to automate the code review process. Instead of replacing human reviewers, it acts as a "First Pass" filter. It connects to the GitHub API, fetches Pull Requests, analyzes code changes for security vulnerabilities and style issues, and drafts constructive comments.

Crucially, it utilizes a Human-in-the-Loop workflow: the agent proposes comments, but the human user must approve them before they are posted to GitHub, preventing AI hallucinations from spamming real repositories.

## Reason for picking up this project

This project demonstrates the practical application of Large Language Models in software engineering workflows. It aligns with the course by:

- **LangGraph:** Managing the cyclic workflow of Fetch -> Analyze -> Review -> Post.

- **Tool Use:** Integrating with external APIs (GitHub) to fetch real-world data.

- **Structured Output:** Ensuring the AI generates parseable JSON comments (Line Number + Body) rather than free text.

- **Human-in-the-Loop:** Implementing a safety layer before writing to a public API.


## Plan

I plan to excecute these steps to complete my project.

- [X] **Step 1:** Setup & State Definition:
Define the ReviewState using Pydantic. This includes schemas for PR metadata, file diffs, and the list of generated review comments.
- [X] **Step 2:** GitHub Integration (Tools):
Implement the GitHubTool class using PyGithub. Create functions to fetch PR diffs and post comments to real repositories.
- [X] **Step 3:** Reviewer Agent:
Build the core analysis node. This uses gpt-4o-mini with a specialized prompt to detect bugs, security flaws, and style violations in the diffs.
- [X] **Step 4:** HITL & Graph Orchestration:
Construct the LangGraph workflow. Implement interrupt_before logic to allow the user to edit/reject AI comments before they are posted.
- [X] **Step 5:** Testing & Validation:
Run the agent against a real "Test Pull Request" to verify it detects issues and posts comments correctly.

## Conclusion:

I had planned to achieve the agent. I think I have achieved the conclusion satisfactorily.

In this capstone, I successfully built GitGuard AI, a functioning autonomous code review assistant. The system moves beyond simple scripts by implementing a stateful, cyclic workflow that mimics a human engineer's process: reading code, thinking about issues, drafting comments, and waiting for final approval before submitting.

I used the following technologies in my project:
- **LangGraph:** I used it to define the StateGraph, manage the flow between the reviewer and poster nodes, and handle interrupt_before logic for human intervention.
- **PyGithub:** I used to interface with the real world. It allows our agent to fetch raw diffs and post Review Comments directly to GitHub Pull Requests.
- **OpenAI:** I utilized this model for its balance of speed and reasoning capability, optimizing the prompt for security and code style analysis.
- **Pydantic:** Essential for Structured Output. I defined strict schemas `PullRequestComment` to ensure the LLM never outputs unstructured text, guaranteeing that every comment has a valid file path and line number.
- **SQLite:** Implemented via `SqliteSaver` to provide Persistence. This allows the graph to "pause" execution, wait for user input indefinitely, and resume exactly where it left off.

I used the course knowledge in the following ways:
- **Human-in-the-Loop (HITL):** I implemented a safety check using LangGraph's checkpointing system.
- **Tool Calling:** I gave it real tools (`fetch_pr_diff`, `post_pr_review`) to perform actual work.
- **State Management:** I implemented a shared `ReviewState`, allowing different parts of the application (Agent vs. Tool) to read and write to a common memory structure.

Project Structure:

![Graph](https://github.com/Nithurshen/nithurshen-capstone-MAT496/blob/main/gitguard_workflow.png)

GitGuard AI solves a genuine problem in software development: **Review Fatigue**.
- **First-Pass Filter**: It catches trivial issues (typos, missing docstrings, minor bugs) automatically, allowing human reviewers to focus on complex architectural logic.
- **Standardization:** It enforces coding standards consistently across a team, without personal bias.
- **Security:** It acts as an automated security auditor, flagging potential vulnerabilities (like secrets in code or injection risks) before they merge.

This project serves as a foundational template for any "AI Co-pilot" application that requires reading external data, making decisions, and acting on the world with human supervision.

A sample run of the project:

```commandline
/Users/nithurshen/SNU/nithurshen-capstone-MAT496/.venv/bin/python /Users/nithurshen/SNU/nithurshen-capstone-MAT496/main.py 
⚠️  Configuration needed.
Enter Repo (format: owner/name): aeon-toolkit/aeon
Enter PR Number: 3133
🚀 Starting GitGuard AI for PR #3133 in aeon-toolkit/aeon...
   Thread ID: 72d88ab7-0f34-4b1d-b834-8a05c49f28d1

--- Phase 1: Fetching & Analyzing ---
🤖 Node 'reviewer' executed.
🤖 Node '__interrupt__' executed.

⏸️  Workflow Paused for Human Review

📝 Proposed 4 Comments:
1. [Minor] aeon/transformations/series/smoothing/_loess.py:1
   "Consider adding a more detailed module docstring that includes usage examples and potential edge cases for better documentation."
----------------------------------------
2. [Minor] aeon/transformations/series/smoothing/_loess.py:10
   "The class docstring should include a description of the expected input types and shapes for better clarity."
----------------------------------------
3. [Nitpick] aeon/transformations/series/smoothing/_loess.py:36
   "The check for `self.degree` could be improved by using a set for valid degrees, which would make it easier to extend in the future if needed."
----------------------------------------
4. [Nitpick] aeon/transformations/series/smoothing/_loess.py:66
   "Consider using `np.clip` for the weights to ensure they remain within valid bounds, which can prevent potential numerical issues."
----------------------------------------

Approve posting these comments? (yes/no): yes

--- Phase 3: Posting to GitHub ---
🤖 Node 'poster' executed.
   Message: 🚀 Review submitted successfully.

Process finished with exit code 0
```

![SS1](https://github.com/Nithurshen/nithurshen-capstone-MAT496/blob/main/ss1.png)

![SS2](https://github.com/Nithurshen/nithurshen-capstone-MAT496/blob/main/ss2.png)

*I have deleted the comments now, as it was a real PR that I contributed to an open source project*