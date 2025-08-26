import os
import nest_asyncio
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# LangChain + NeMo Guardrails
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from nemoguardrails import LLMRails, RailsConfig

# --- Setup ---
nest_asyncio.apply()
load_dotenv()  # loads OPENAI_API_KEY and NVIDIA_API_KEY from .env

# --- NVIDIA LLM ---
llm = ChatNVIDIA(
    model="deepseek-ai/deepseek-r1",
    api_key=os.getenv("NVIDIA_API_KEY"),  # ✅ use env var
    temperature=0.6,
    top_p=0.7,
    max_tokens=4096
)

# --- Load Guardrails Config ---
config = RailsConfig.from_path("config")

# --- Create Guardrails Wrapper ---
app_guardrails = LLMRails(config, verbose=True, llm=llm)

# --- Flask App ---
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    response = app_guardrails.generate(messages=[{"role": "user", "content": user_message}])

    # Extract content if dict
    if isinstance(response, dict) and "content" in response:
        raw_content = response["content"]
    else:
        raw_content = str(response)

    # --- Only return Bot message ---
    import re
    match = re.search(r'\*\*Bot message:\*\*\s*"([^"]+)"', raw_content)
    if match:
        bot_message = match.group(1)
    else:
        # fallback → return raw text if no Bot message found
        bot_message = raw_content

    return jsonify({"response": bot_message})


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Chatbot API is running 🚀"})


# --- Run Flask ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
