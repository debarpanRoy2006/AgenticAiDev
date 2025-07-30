import json
import requests
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import datetime
import base64
import uuid # For generating unique IDs for ChromaDB documents

# ChromaDB and Sentence-Transformers imports
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# Initialize ChromaDB client and embedding model
# For a persistent local client (data is stored in a file in your project root):
# Data will be stored in a folder named 'chroma_db_data' next to your manage.py
chroma_client = chromadb.PersistentClient(path="./chroma_db_data") 
# For an in-memory client (data is lost on restart - good for quick demos without disk persistence):
# chroma_client = chromadb.Client() # COMMENT OUT THIS LINE IF USING PERSISTENT CLIENT


# Initialize embedding function for ChromaDB
# This will download the model 'all-MiniLM-L6-v2' if not already present
chroma_embedding_function = None
try:
    chroma_embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    print("SentenceTransformerEmbeddingFunction initialized.")
except Exception as e:
    print(f"Error initializing ChromaDB embedding function: {e}")


# Get or create a collection
collection_name = "agent_ai_interactions"
vector_db_collection = None # Initialize to None
if chroma_embedding_function: # Only try to get/create if embedding function loaded
    try:
        # CORRECTED: Pass the embedding_function directly when getting/creating the collection
        vector_db_collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=chroma_embedding_function 
        )
        print(f"ChromaDB collection '{collection_name}' initialized successfully.")
    except Exception as e:
        print(f"Error getting/creating ChromaDB collection: {e}")
else:
    print("ChromaDB embedding function not initialized. VectorDB memory will be unavailable.")


# Initialize embedding model for direct use (e.g., for _get_conversation_context query)
# This is separate from the one passed to ChromaDB collection, but uses the same model.
embedding_model = None
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("SentenceTransformer model for direct use initialized.")
except Exception as e:
    print(f"Error loading SentenceTransformer model for direct use: {e}")


class AgentCore:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    def _get_conversation_context(self, current_prompt_text, num_results=3):
        """
        Retrieves semantically relevant past interactions from ChromaDB
        and formats them as a string for LLM context.
        """
        context_str = ""
        # Check if embedding_model and vector_db_collection are successfully initialized
        if not current_prompt_text or not embedding_model or not vector_db_collection:
            print("Skipping context retrieval: Missing prompt, embedding model, or vector DB collection.")
            return context_str

        try:
            # Generate embedding for the current prompt using the direct embedding_model
            query_embedding = embedding_model.encode(current_prompt_text).tolist()

            # Query ChromaDB for similar documents
            results = vector_db_collection.query(
                query_embeddings=[query_embedding],
                n_results=num_results,
                include=['documents', 'metadatas', 'distances']
            )

            if results and results['documents'] and results['documents'][0]:
                context_str += "--- Relevant Past Conversation Context (from memory) ---\n"
                for i in range(len(results['documents'][0])):
                    doc_content = results['documents'][0][i]
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]

                    # You might want to filter by a similarity threshold here
                    # For SentenceTransformers, lower distance means more similar.
                    # A common threshold for cosine similarity (1-distance) might be 0.7 or higher.
                    # if (1 - distance) < 0.7: continue 

                    context_str += (
                        f"Interaction (Similarity: {1-distance:.2f}, Type: {metadata.get('action_type', 'N/A')}):\n"
                        f"  User Prompt: {metadata.get('prompt', 'N/A')}\n"
                    )
                    if metadata.get('file_uploaded'):
                        context_str += f"  File Uploaded: Yes (Type: {metadata.get('file_type', 'N/A')})\n"
                    context_str += f"  AI Response: {doc_content[:100]}...\n" # Use full doc_content for AI response
                context_str += "---------------------------------------------------\n\n"
        except Exception as e:
            print(f"Error retrieving context from ChromaDB: {e}")
            context_str = ""
        return context_str

    def _call_gemini(self, prompt_parts, generation_config=None, include_context=True, current_prompt_text_for_context=""):
        """
        Helper to call the Gemini API, now accepting a list of prompt parts (text and/or inlineData).
        Includes conversation context from ChromaDB based on current_prompt_text_for_context.
        """
        headers = {
            'Content-Type': 'application/json',
        }
        params = {'key': self.api_key}

        final_prompt_parts = []
        if include_context and current_prompt_text_for_context:
            context_text = self._get_conversation_context(current_prompt_text_for_context)
            if context_text:
                final_prompt_parts.append({"text": context_text})

        final_prompt_parts.extend(prompt_parts) # Add the current prompt parts

        payload = {
            "contents": [{"role": "user", "parts": final_prompt_parts}]
        }
        if generation_config:
            payload["generationConfig"] = generation_config

        try:
            response = requests.post(self.base_url, headers=headers, params=params, json=payload)
            response.raise_for_status()
            result = response.json()

            if result.get('candidates') and result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts'):
                text_content = result['candidates'][0]['content']['parts'][0]['text']
                if generation_config and generation_config.get('responseMimeType') == 'application/json':
                    try:
                        return json.loads(text_content)
                    except json.JSONDecodeError:
                        return text_content # Return raw text if JSON parsing fails
                return text_content
            else:
                return "No valid response from LLM."
        except requests.exceptions.RequestException as e:
            print(f"Error calling Gemini API: {e}")
            return f"Error communicating with AI: {e}"
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return f"An unexpected error occurred: {e}"

    def planning_module(self, prompt, file_content=None, file_data=None, file_type=None, action_type_from_frontend=None):
        """
        MCP Component: Planning Module
        Determines action based on explicit frontend action_type or prompt and file type.
        """
        prompt_lower = prompt.lower()

        # IMPORTANT: Prioritize explicit action_type from frontend first
        if action_type_from_frontend == "code_generation":
            return {"action_type": "code_generation", "task_description": prompt, "file_content": file_content}
        elif action_type_from_frontend == "debugging":
            return {"action_type": "debugging", "task_description": prompt, "file_content": file_content}
        elif action_type_from_frontend == "git_operation":
            return {"action_type": "git_operation", "task_description": prompt, "file_content": file_content}
        elif action_type_from_frontend == "analyze_file":
            return {"action_type": "analyze_file", "task_description": prompt, "file_content": file_content, "file_type": file_type}
        elif action_type_from_frontend == "analyze_image":
            return {"action_type": "analyze_image", "task_description": prompt, "file_data": file_data, "file_type": file_type}
        elif action_type_from_frontend == "analyze_document":
            return {"action_type": "analyze_document", "task_description": prompt, "file_content": file_content, "file_type": file_type}
        elif action_type_from_frontend == "generate_ideas":
            return {"action_type": "generate_ideas", "task_description": prompt, "file_content": file_content}
        elif action_type_from_frontend == "general_ai":
            return {"action_type": "general_ai", "task_description": prompt, "file_content": file_content, "file_data": file_data, "file_type": file_type}
        
        # Fallback to keyword-based planning if no explicit action_type_from_frontend
        # This part is less critical now but good for robustness if frontend doesn't send action_type
        if file_data and file_type and file_type.startswith('image/'):
            return {"action_type": "analyze_image", "task_description": prompt, "file_data": file_data, "file_type": file_type}
        elif file_content and (file_type == 'application/pdf' or file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'):
            return {"action_type": "analyze_document", "task_description": prompt, "file_content": file_content, "file_type": file_type}
        elif file_content: # Generic text file analysis
            return {"action_type": "analyze_file", "task_description": prompt, "file_content": file_content, "file_type": file_type}

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
            return {"action_type": "general_ai", "task_description": prompt}


    def tool_executor(self, action_plan):
        """
        MCP Component: Tool Executor
        Dispatches the task to the appropriate tool based on the action plan.
        """
        action_type = action_plan.get("action_type")
        task_description = action_plan.get("task_description")
        file_content = action_plan.get("file_content")
        file_data = action_plan.get("file_data")
        file_type = action_plan.get("file_type")

        if action_type == "code_generation":
            return self._generate_code_tool(task_description, file_content)
        elif action_type == "debugging":
            return self._debug_code_tool(task_description, file_content)
        elif action_type == "git_operation":
            return self._execute_git_command_tool(task_description, file_content)
        elif action_type == "analyze_file":
            return self._analyze_file_tool(task_description, file_content)
        elif action_type == "analyze_image":
            return self._analyze_image_tool(task_description, file_data, file_type)
        elif action_type == "analyze_document":
            return self._analyze_document_tool(task_description, file_content, file_type)
        elif action_type == "generate_ideas":
            return self._generate_ideas_tool(task_description, file_content)
        elif action_type == "general_ai":
            return self._general_purpose_ai_tool(task_description, file_content, file_data, file_type)
        else:
            return "Unknown action type requested by the planning module."

    def _generate_code_tool(self, prompt, file_content=None):
        llm_prompt_parts = [{"text": f"Generate code based on the following request. Provide only the code, no explanations:\n{prompt}"}]
        if file_content:
            llm_prompt_parts.append({"text": f"\n\nConsider this file content for context or modification:\n```\n{file_content}\n```"})
        return self._call_gemini(llm_prompt_parts, current_prompt_text_for_context=prompt)

    def _debug_code_tool(self, prompt, file_content=None):
        llm_prompt_parts = [{"text": f"Analyze the following code/error and suggest debugging steps or corrections. Be concise:\n"}]
        if file_content:
            llm_prompt_parts.append({"text": f"Code from file:\n```\n{file_content}\n```\n"})
        llm_prompt_parts.append({"text": f"User prompt/additional context: {prompt}"})
        return self._call_gemini(llm_prompt_parts, current_prompt_text_for_context=prompt)

    def _execute_git_command_tool(self, prompt, file_content=None):
        llm_prompt_parts = [{"text": f"Based on the following request, provide a simulated outcome for a Git command. Do not actually execute it. For example, if asked to 'commit changes', respond with 'Simulated: Changes committed successfully with message...'. Request: {prompt}"}]
        if file_content:
            llm_prompt_parts.append({"text": f"\n\nFile content (if relevant for context):\n```\n{file_content}\n```"})
        return self._call_gemini(llm_prompt_parts, current_prompt_text_for_context=prompt)

    def _analyze_file_tool(self, prompt, file_content):
        if not file_content:
            return "No text file content provided for analysis."
        llm_prompt_parts = [{"text": f"Analyze the following text file content. {prompt if prompt else 'Provide a summary or key insights.'}:\n```\n{file_content}\n```"}]
        return self._call_gemini(llm_prompt_parts, current_prompt_text_for_context=prompt)

    def _analyze_image_tool(self, prompt, image_data, image_type):
        if not image_data or not image_type:
            return "No image data provided for analysis."

        llm_prompt_parts = [
            {"text": prompt if prompt else "Describe this image in detail."},
            {"inlineData": {"mimeType": image_type, "data": image_data}}
        ]
        return self._call_gemini(llm_prompt_parts, current_prompt_text_for_context=prompt)

    def _analyze_document_tool(self, prompt, file_content, file_type):
        if not file_content:
            return f"No content extracted from {file_type} for analysis."
        
        llm_prompt_parts = [{"text": f"Analyze the content of this document (type: {file_type}). {prompt if prompt else 'Provide a summary or key insights.'}:\n```\n{file_content}\n```"}]
        return self._call_gemini(llm_prompt_parts, current_prompt_text_for_context=prompt)


    def _generate_ideas_tool(self, prompt, file_content=None):
        llm_prompt = (
            f"Generate a list of 5-7 creative ideas for blog posts, articles, or website topics based on the following request.\n"
            f"Provide ONLY a JSON array of strings, where each string is a single idea.\n"
            f"Do NOT include any introductory text, concluding remarks, markdown code blocks (```json), or any other text outside the pure JSON array.\n"
            f"Example: [\"Idea 1\", \"Idea 2\", \"Idea 3\"]\n"
            f"Request: {prompt}"
        )
        if file_content:
            llm_prompt += f"\n\nConsider this file content for additional context:\n```\n{file_content}\n```"
        
        generation_config = {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "ARRAY",
                "items": {
                    "type": "STRING"
                }
            }
        }
        return self._call_gemini([{"text": llm_prompt}], generation_config=generation_config, current_prompt_text_for_context=prompt)

    def _general_purpose_ai_tool(self, prompt, file_content=None, file_data=None, file_type=None):
        llm_prompt_parts = [
            {"text": (
                f"You are a general-purpose AI assistant. Provide concise and informative answers to the user's request, like a human. "
                f"Your primary goal is to answer questions concisely and informatively, like a human. "
                f"Crucially, you MUST NOT generate any code, scripts, programming examples, or technical solutions. "
                f"If the user asks for anything related to code (e.g., 'write code', 'debug this', 'how to implement X in Python'), "
                f"politely tell them that you are a general AI and cannot generate code. "
                f"Instead, instruct them to close this modal and use the dedicated 'Generate Code' or 'Debugging' tools on the main page. "
                f"Similarly, if they ask for Git operations, tell them to use the 'Git Operation' tool. "
                f"If they ask for file analysis, suggest the 'Analyze File' tool. "
                f"If they ask for ideas, suggest the 'Generate Ideas' tool.\n\n"
                f"User Request: {prompt}"
            )}
        ]
        if file_data and file_type and file_type.startswith('image/'):
            llm_prompt_parts.append({"inlineData": {"mimeType": file_type, "data": file_data}})
        elif file_content:
            llm_prompt_parts.append({"text": f"\n\nAdditional context from file (type: {file_type}):\n```\n{file_content}\n```"})
            
        return self._call_gemini(llm_prompt_parts, current_prompt_text_for_context=prompt)


    def memory_manager(self, interaction_log):
        content_to_embed = f"User: {interaction_log.get('prompt', '')}\nAI: {interaction_log.get('result', '')}"
        
        if embedding_model and vector_db_collection:
            try:
                embedding = embedding_model.encode(content_to_embed).tolist()

                # Ensure all metadata values are non-None and of correct type for ChromaDB
                metadata_to_store = {
                    "prompt": interaction_log.get('prompt', '') or 'N/A', # Ensure string, not None
                    "action_type": interaction_log.get('action_type', '') or 'N/A', # Ensure string, not None
                    "file_uploaded": bool(interaction_log.get('file_uploaded', False)), # Ensure boolean
                    "file_type": interaction_log.get('file_type', '') or 'N/A', # Ensure string, not None
                    "result": interaction_log.get('result', '') or 'N/A', # Ensure string, not None
                    "timestamp": interaction_log.get('timestamp', '') or 'N/A' # Ensure string, not None
                }

                vector_db_collection.add(
                    documents=[content_to_embed],
                    metadatas=[metadata_to_store], # Use the cleaned metadata
                    ids=[str(uuid.uuid4())],
                    embeddings=[embedding]
                )
                print(f"Added interaction to ChromaDB. Total docs: {vector_db_collection.count()}")
            except Exception as e:
                print(f"Error adding to ChromaDB: {e}")
        else:
            print("ChromaDB client or embedding model not initialized. Skipping memory storage.")


agent = AgentCore(api_key=settings.GEMINI_API_KEY)

from django.shortcuts import render

def home_view(request):
    return render(request, 'index.html')

@csrf_exempt
@require_http_methods(["POST"])
def agent_api_view(request):
    try:
        action_type_from_frontend = request.POST.get('action_type')
        prompt = request.POST.get('prompt', '').strip()
        
        uploaded_file_data = request.POST.get('uploaded_file_data')
        uploaded_file_type = request.POST.get('uploaded_file_type')

        uploaded_file = request.FILES.get('uploaded_file')

        file_content = None
        file_data_for_llm = None
        file_type_for_llm = None

        if uploaded_file:
            if uploaded_file.size > 2 * 1024 * 1024:
                return JsonResponse({"error": "File size exceeds 2MB limit."}, status=400)
            
            file_type_for_llm = uploaded_file.content_type
            
            if file_type_for_llm and file_type_for_llm.startswith('image/'):
                file_data_for_llm = base64.b64encode(uploaded_file.read()).decode('utf-8')
            elif file_type_for_llm == 'application/pdf' or file_type_for_llm == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                try:
                    file_content = uploaded_file.read().decode('utf-8')
                except UnicodeDecodeError:
                    file_content = f"Binary content of {uploaded_file.name}. (Text extraction not implemented for this file type in demo)"
            else:
                try:
                    file_content = uploaded_file.read().decode('utf-8')
                except UnicodeDecodeError:
                    file_content = f"Could not decode text file {uploaded_file.name}. Ensure it's valid UTF-8."
        
        elif uploaded_file_data and uploaded_file_type:
            file_data_for_llm = uploaded_file_data
            file_type_for_llm = uploaded_file_type


        if action_type_from_frontend in ['generate_ideas', 'general_ai']:
            if not prompt:
                return JsonResponse({"error": "Prompt is required for this action."}, status=400)
        else:
            if action_type_from_frontend == 'analyze_image' and not file_data_for_llm:
                 return JsonResponse({"error": "Image file is required for image analysis."}, status=400)
            elif (action_type_from_frontend in ['analyze_file', 'analyze_document', 'code_generation', 'debugging', 'git_operation']) and not prompt and not file_content:
                return JsonResponse({"error": "Prompt or text file content is required for this action."}, status=400)


        valid_action_types = [
            "code_generation", "debugging", "git_operation", "analyze_file",
            "generate_ideas", "general_ai", "analyze_image", "analyze_document"
        ]
        if action_type_from_frontend not in valid_action_types:
            return JsonResponse({"error": "Invalid action type provided."}, status=400)

        action_plan = agent.planning_module(prompt, file_content, file_data_for_llm, file_type_for_llm, action_type_from_frontend) # Pass action_type_from_frontend

        result = agent.tool_executor(action_plan)

        agent.memory_manager({
            "prompt": prompt,
            "action_type": action_type_from_frontend,
            "file_uploaded": bool(uploaded_file or uploaded_file_data),
            "file_type": file_type_for_llm, # This can be None, handled in memory_manager
            "result": result,
            "timestamp": datetime.datetime.now().isoformat()
        })

        return JsonResponse({"result": result})

    except Exception as e:
        print(f"Server error: {e}")
        return JsonResponse({"error": f"Internal server error: {e}"}, status=500)


@require_http_methods(["GET"])
def get_history_view(request):
    try:
        # Ensure vector_db_collection is initialized before trying to use it
        if not vector_db_collection:
            return JsonResponse({"history": []}, status=200) # Return empty history if DB not ready

        # Get all IDs from the collection
        all_ids = vector_db_collection.get()['ids']
        if not all_ids:
            return JsonResponse({"history": []}, status=200) # No documents yet

        all_docs = vector_db_collection.get(
            ids=all_ids,
            include=['metadatas', 'documents']
        )
        
        history_list = []
        if all_docs and all_docs['ids']:
            for i in range(len(all_docs['ids'])):
                metadata = all_docs['metadatas'][i]
                document_content = all_docs['documents'][i]
                
                history_list.append({
                    "prompt": metadata.get('prompt', 'N/A'),
                    "action_type": metadata.get('action_type', 'N/A'),
                    "file_uploaded": metadata.get('file_uploaded', False),
                    "file_type": metadata.get('file_type', 'N/A'),
                    "result": metadata.get('result', document_content), # Prefer metadata result, fallback to doc content
                    "timestamp": metadata.get('timestamp', 'N/A')
                })
        
        # Sort history by timestamp if needed (ChromaDB doesn't guarantee order)
        # Ensure timestamp is comparable (e.g., ISO format string)
        history_list.sort(key=lambda x: x.get('timestamp', ''), reverse=False) # Sort ascending by time

        return JsonResponse({"history": history_list})
    except Exception as e:
        print(f"Error fetching history from ChromaDB: {e}")
        return JsonResponse({"error": f"Internal server error fetching history: {e}"}, status=500)
