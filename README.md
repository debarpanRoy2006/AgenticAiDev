# Agentic AI Development Assistant



A powerful web-based assistant designed to streamline software development tasks using Agentic AI principles and the Gemini LLM. It features capabilities for code generation, debugging, Git operation simulation, file analysis, idea generation, and a general-purpose AI chat.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Project Phases](#project-phases)
- [Contributing](#contributing)
- [License](#license)
- [Team](#team)

## Features

- **Code Generation:** Generate code snippets or functions based on natural language prompts.
- **Code Debugging:** Get suggestions for fixing errors in provided code.
- **Git Operation Simulation:** Simulate common Git commands and their outcomes.
- **File Analysis:** Upload and analyze text-based files (code, markdown, etc.) for summaries or insights.
- **Idea Generation:** Brainstorm creative ideas (e.g., blog posts, articles) based on keywords, presented in a structured list.
- **General Purpose AI Chat:** A dedicated chat interface for general questions, strictly avoiding code generation and redirecting to specialized tools when appropriate.
- **Conversation History:** A sidebar displaying past interactions, allowing users to revisit previous queries and results.
- **Responsive UI:** Optimized for various screen sizes, including a collapsible sidebar/navbar for mobile.

## Architecture

This project implements an **MCP (Multi-Component Processing)** server architecture, simulating distinct, specialized components for different AI functionalities.

### Key Components:

- **Frontend (Web Interface):**
    - Provides the user interface for interaction.
    - Sends user requests (prompts, files, action types) to the backend.
    - Displays AI responses and manages conversation history.
- **Backend (Django Server):**
    - Acts as the central orchestrator and API endpoint.
    - Contains the core **Agentic AI Logic** (`AgentCore` class).
    - **Planning Module:** Interprets user intent and determines the appropriate action/tool.
    - **Tool Executor:** Dispatches tasks to specialized AI tools.
    - **Memory Manager:** Stores and retrieves conversation history for context.
- **AI Tools (within Backend):**
    - `_generate_code_tool`: Interacts with LLM for code generation.
    - `_debug_code_tool`: Interacts with LLM for debugging suggestions.
    - `_execute_git_command_tool`: Interacts with LLM for simulated Git outcomes.
    - `_analyze_file_tool`: Interacts with LLM for file content analysis.
    - `_generate_ideas_tool`: Interacts with LLM for structured idea generation.
    - `_general_purpose_ai_tool`: Interacts with LLM for general information, with strict no-code constraints.
- **External LLM (Gemini AI):**
    - Provides the underlying intelligence for all AI functionalities via API calls.

### Communication Protocols:

- **Frontend ↔ Backend:** HTTP POST requests (for actions) and HTTP GET requests (for history) using `multipart/form-data` (for files) and JSON.
- **Backend ↔ LLM:** HTTPS POST requests with JSON payloads to the Google Gemini API.

## Tech Stack

- **Frontend:** HTML5, CSS3 (Bootstrap 5), JavaScript
- **Backend:** Python 3.x, Django
- **LLM API:** Google Gemini 2.0 Flash
- **Dependencies:** `django`, `django-cors-headers`, `requests`

## Setup and Installation

Follow these steps to get the project running on your local machine.

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
