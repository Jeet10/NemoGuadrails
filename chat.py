import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from nemoguardrails import LLMRails, RailsConfig

# --- Load environment ---
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")

# --- Define NVIDIA LLM ---
main_llm = ChatNVIDIA(
    model="deepseek-ai/deepseek-r1",
    api_key=api_key,
    temperature=0.6,
    top_p=0.7,
    max_completion_tokens=4096,
)

# --- Load NeMo Guardrails config ---
config = RailsConfig.from_path("config")
rails = LLMRails(config, llm=main_llm, verbose=True)

# --- Flask app ---
app = Flask(__name__)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id")

    if not user_message:
        return jsonify({"error": "Empty message."}), 400

    messages = [{"role": "user", "content": user_message}]

    try:
        result = rails.generate(messages=messages)

        # ✅ Correct extraction of reply
        assistant_reply = result.get("content", "")

        return jsonify({
            "reply": assistant_reply,
            "used_llm": True,
            "session_id": session_id
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "session_id": session_id
        }), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
