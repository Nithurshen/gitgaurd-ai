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
- [ ] **Step 3:** The Reviewer Agent:
Build the core analysis node. This uses gpt-4o-mini with a specialized prompt to detect bugs, security flaws, and style violations in the diffs.
- [ ] **Step 4:** HITL & Graph Orchestration:
Construct the LangGraph workflow. Implement interrupt_before logic to allow the user to edit/reject AI comments before they are posted.
- [ ] **Step 5:** Testing & Validation:
Run the agent against a real "Test Pull Request" to verify it detects issues and posts comments correctly.

## Conclusion:

I had planned to achieve {this this}. I think I have/have-not achieved the conclusion satisfactorily. The reason for your satisfaction/unsatisfaction.
