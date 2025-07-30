import json
import requests
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import datetime # Import for timestamp

# Global variable to simulate agent memory (for demo purposes)
# In a real application, this would be a database (e.g., Vector DB)
agent_memory = [] # Stores a list of dictionaries, each representing an interaction

class AgentCore:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    def _get_conversation_context(self, num_interactions=3):
        """
        Retrieves the last 'num_interactions' from agent_memory
        and formats them as a string for LLM context.
        """
        context_str = ""
        if agent_memory:
            recent_interactions = agent_memory[-num_interactions:] # Get last N interactions
            context_str += "--- Past Conversation Context ---\n"
            for i, interaction in enumerate(recent_interactions):
                context_str += f"Interaction {i+1} ({interaction.get('timestamp', 'N/A')}):\n"
                context_str += f"  User Prompt: {interaction.get('prompt', 'N/A')}\n"
                if interaction.get('file_uploaded'):
                    context_str += f"  File Uploaded: Yes\n"
                context_str += f"  AI Response: {interaction.get('result', 'N/A')[:100]}...\n" # Truncate for brevity
            context_str += "-------------------------------\n\n"
        return context_str

    def _call_gemini(self, prompt, generation_config=None, include_context=True):
        """
        Helper to call the Gemini API, now optionally including past conversation context.
        """
        headers = {
            'Content-Type': 'application/json',
        }
        params = {'key': self.api_key}

        full_prompt = ""
        if include_context:
            full_prompt += self._get_conversation_context()

        full_prompt += prompt # Append the current prompt

        payload = {
            "contents": [{"role": "user", "parts": [{"text": full_prompt}]}]
        }
        if generation_config:
            payload["generationConfig"] = generation_config

        try:
            response = requests.post(self.base_url, headers=headers, params=params, json=payload)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            result = response.json()

            if result.get('candidates') and result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts'):
                text_content = result['candidates'][0]['content']['parts'][0]['text']
                try:
                    # Attempt to parse as JSON if a schema was likely used
                    if generation_config and generation_config.get('responseMimeType') == 'application/json':
                        return json.loads(text_content)
                except json.JSONDecodeError:
                    # If JSON parsing fails, return the raw text content
                    return text_content
                return text_content
            else:
                return "No valid response from LLM."
        except requests.exceptions.RequestException as e:
            print(f"Error calling Gemini API: {e}")
            return f"Error communicating with AI: {e}"
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return f"An unexpected error occurred: {e}"

    def planning_module(self, prompt, file_content=None, file_data=None, file_type=None):
            """
            MCP Component: Planning Module
            Determines action based on prompt and file type.
            """
            prompt_lower = prompt.lower()

            # Prioritize actions based on file type if present
            if file_data and file_type and file_type.startswith('image/'):
                return {"action_type": "analyze_image", "task_description": prompt, "file_data": file_data, "file_type": file_type}
            elif file_content and (file_type == 'application/pdf' or file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'):
                # For PDF/Word, we'll treat it as text analysis for demo.
                # In real app, you'd extract text here using a library.
                return {"action_type": "analyze_document", "task_description": prompt, "file_content": file_content, "file_type": file_type}
            elif file_content: # Generic text file analysis
                return {"action_type": "analyze_file", "task_description": prompt, "file_content": file_content, "file_type": file_type}

            # Then prioritize actions based on prompt keywords
            if "general ai" in prompt_lower or "what is" in prompt_lower or "explain" in prompt_lower or "tell me about" in prompt_lower or "define" in prompt_lower:
                return {"action_type": "general_ai", "task_description": prompt}
            elif "generate code" in prompt_lower or "write code" in prompt_lower or "code for" in prompt_lower:
                return {"action_type": "code_generation", "task_description": prompt}
            elif "debug" in prompt_lower or "fix this code" in prompt_lower or "error in" in prompt_lower:
                return {"action_type": "debugging", "task_description": prompt}
            elif "git" in prompt_lower or "commit" in prompt_lower or "push" in prompt_lower or "pull" in prompt_lower:
                return {"action_type": "git_operation", "task_description": prompt}
            elif "generate ideas" in prompt_lower or "ideas for" in prompt_lower or "brainstorm" in prompt_lower:
                return {"action_type": "generate_ideas", "task_description": prompt}
            else:
                # Default to general_ai if no specific action is detected and no file is provided
                return {"action_type": "general_ai", "task_description": prompt}

    def tool_executor(self, action_plan):
        """
        MCP Component: Tool Executor
        Dispatches the task to the appropriate tool based on the action plan.
        """
        action_type = action_plan.get("action_type")
        task_description = action_plan.get("task_description")
        file_content = action_plan.get("file_content")

        if action_type == "code_generation":
            return self._generate_code_tool(task_description, file_content)
        elif action_type == "debugging":
            return self._debug_code_tool(task_description, file_content)
        elif action_type == "git_operation":
            return self._execute_git_command_tool(task_description, file_content)
        elif action_type == "analyze_file":
            return self._analyze_file_tool(task_description, file_content)
        elif action_type == "generate_ideas": # New tool execution path
            return self._generate_ideas_tool(task_description, file_content)
        elif action_type == "general_ai": # New tool execution path
            return self._general_purpose_ai_tool(task_description, file_content)
        else:
            return "Unknown action type requested by the planning module."

    def _generate_code_tool(self, prompt, file_content=None):
        """
        Tool: Code Generation
        Uses LLM to generate code based on the prompt.
        Can optionally consider file_content for context.
        """
        llm_prompt = f"Generate code based on the following request. Provide only the code, no explanations:\n{prompt}"
        if file_content:
            llm_prompt += f"\n\nConsider this file content for context or modification:\n```\n{file_content}\n```"
        return self._call_gemini(llm_prompt)

    def _debug_code_tool(self, prompt, file_content=None):
        """
        Tool: Debugging
        Uses LLM to suggest debugging steps or corrections for code.
        Can debug code provided in the prompt or from the file_content.
        """
        llm_prompt = f"Analyze the following code/error and suggest debugging steps or corrections. Be concise:\n"
        if file_content:
            llm_prompt += f"Code from file:\n```\n{file_content}\n```\n"
        llm_prompt += f"User prompt/additional context: {prompt}"
        return self._call_gemini(llm_prompt)

    def _execute_git_command_tool(self, prompt, file_content=None):
        """
        Tool: Git Operation (Simulated)
        Uses LLM to interpret a Git-related request and provides a simulated response.
        """
        llm_prompt = f"Based on the following request, provide a simulated outcome for a Git command. Do not actually execute it. For example, if asked to 'commit changes', respond with 'Simulated: Changes committed successfully with message...'. Request: {prompt}"
        if file_content:
            llm_prompt += f"\n\nFile content (if relevant for context):\n```\n{file_content}\n```"
        return self._call_gemini(llm_prompt)

    def _analyze_file_tool(self, prompt, file_content):
        """
        Tool: File Analysis
        Uses LLM to analyze the content of an uploaded file based on the prompt.
        """
        if not file_content:
            return "No file content provided for analysis."

        llm_prompt = f"Analyze the following file content. {prompt if prompt else 'Provide a summary or key insights.'}:\n```\n{file_content}\n```"
        return self._call_gemini(llm_prompt)

    def _generate_ideas_tool(self, prompt, file_content=None):
        """
        New Tool: Generate Ideas
        Uses LLM to generate creative ideas based on the prompt and optional file content.
        Returns a structured JSON array of ideas.
        """
        # Modified prompt to be more strict about JSON output
        llm_prompt = f"Generate a list of 5-7 creative ideas for blog posts, articles, or website topics based on the following request. Provide ONLY a JSON array of strings, where each string is a single idea. Do NOT include any introductory text, concluding remarks, markdown code blocks (```json), or any other text outside the pure JSON array. Example: [\"Idea 1\", \"Idea 2\"]\nRequest: {prompt}"
        if file_content:
            llm_prompt += f"\n\nConsider this file content for additional context:\n```\n{file_content}\n```"
        
        # Configure generation to return JSON
        generation_config = {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "ARRAY",
                "items": {
                    "type": "STRING"
                }
            }
        }
        return self._call_gemini(llm_prompt, generation_config=generation_config)

    def _general_purpose_ai_tool(self, prompt, file_content=None):
        """
        New Tool: General Purpose AI
        Uses LLM to provide general information, strictly avoiding code generation.
        Suggests using specific tools if code is requested.
        """
        # Enhanced prompt for General AI
        llm_prompt = (
            f"You are a general-purpose AI assistant. Provide concise and informative answers to the user's request. "
            f"Your primary goal is to answer questions concisely and informatively, like a human. "
            f"Crucially, you MUST NOT generate any code, scripts, programming examples, or technical solutions. "
            f"If the user asks for anything related to code (e.g., 'write code', 'debug this', 'how to implement X in Python'), "
            f"politely tell them that you are a general AI and cannot generate code. "
            f"Instead, instruct them to close this modal and use the dedicated 'Generate Code' or 'Debugging' tools on the main page. "
            f"Similarly, if they ask for Git operations, tell them to use the 'Git Operation' tool. "
            f"If they ask for file analysis, suggest the 'Analyze File' tool. "
            f"If they ask for ideas, suggest the 'Generate Ideas' tool.\n\n"
            f"User Request: {prompt}"
        )
        if file_content:
            llm_prompt += f"\n\nAdditional context from file:\n```\n{file_content}\n```"
        return self._call_gemini(llm_prompt)


    def memory_manager(self, interaction_log):
        """
        MCP Component: Memory Manager
        Stores interaction history.
        """
        agent_memory.append(interaction_log)
        if len(agent_memory) > 10: # Keep memory size manageable for demo
            agent_memory.pop(0)


# Initialize the AgentCore
agent = AgentCore(api_key=settings.GEMINI_API_KEY)

# Django view to render the main HTML template
from django.shortcuts import render

def home_view(request):
    """
    Renders the main index.html template.
    """
    return render(request, 'index.html')

@csrf_exempt
@require_http_methods(["POST"])
def agent_api_view(request):
    """
    Main API endpoint for the Agentic AI Assistant.
    Handles file uploads, prompt, and action type from the frontend.
    """
    try:
        action_type_from_frontend = request.POST.get('action_type')
        prompt = request.POST.get('prompt', '').strip()
        uploaded_file = request.FILES.get('uploaded_file')

        file_content = None
        if uploaded_file:
            try:
                if uploaded_file.size > 2 * 1024 * 1024:  # 2MB limit
                    return JsonResponse({"error": "File size exceeds 2MB limit."}, status=400)
                try:
                    file_content = uploaded_file.read().decode('utf-8')
                except UnicodeDecodeError:
                    file_content = uploaded_file.read().decode('latin-1')
            except Exception as e:
                return JsonResponse({"error": f"Error reading file: {e}"}, status=400)

        # Validation for different action types
        if action_type_from_frontend in ['generate_ideas', 'general_ai']:
            if not prompt:
                return JsonResponse({"error": "Prompt is required for this action."}, status=400)
        else:
            if not prompt and not file_content:
                return JsonResponse({"error": "Prompt or uploaded file is required for this action."}, status=400)

        valid_action_types = [
            "code_generation", "debugging", "git_operation",
            "analyze_file", "generate_ideas", "general_ai"
        ]
        if action_type_from_frontend not in valid_action_types:
            return JsonResponse({"error": "Invalid action type provided."}, status=400)

        # 1. Planning Module
        action_plan = agent.planning_module(prompt, file_content)
        # Override action type with frontend's selection for consistency
        action_plan["action_type"] = action_type_from_frontend

        # 2. Tool Execution
        result = agent.tool_executor(action_plan)

        # 3. Memory Management (Log the interaction)
        agent.memory_manager({
            "prompt": prompt,
            "action_type": action_type_from_frontend,
            "file_uploaded": bool(uploaded_file),
            "result": result,
            "timestamp": datetime.datetime.now().isoformat()
        })

        return JsonResponse({"result": result})

    except Exception as e:
        print(f"Server error: {e}")
        return JsonResponse({"error": f"Internal server error: {e}"}, status=500)
@require_http_methods(["GET"])
def get_history_view(request):
   
    return JsonResponse({"history": agent_memory})

# --- agent_ai_project/settings.py ---
# Add 'agent_app' and 'corsheaders' to INSTALLED_APPS
# Configure CORS_ALLOWED_ORIGINS (adjust for your frontend's URL)
# Add CORS_ALLOW_ALL_ORIGINS = True for development simplicity, but be specific in production.

