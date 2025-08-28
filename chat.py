import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from nemoguardrails import LLMRails

from dynamic_config import build_dynamic_config
from evaluation_utils import run_fact_check_evaluation

# --- Load environment ---
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")
if not api_key:
    raise RuntimeError("NVIDIA_API_KEY environment variable not set.")

# --- Define NVIDIA LLM ---
main_llm = ChatNVIDIA(
    model="deepseek-ai/deepseek-r1",
    api_key=api_key,
    temperature=0.6,
    top_p=0.7,
    max_completion_tokens=4096,
)

# --- Build NeMo Guardrails config dynamically ---
rails_config = build_dynamic_config(api_key)
rails = LLMRails(rails_config, llm=main_llm, verbose=True)

# --- Flask app ---
app = Flask(__name__)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id")

    if not user_message:
        return jsonify({"error": "Empty message."}), 400

    try:
        result = rails.generate(messages=[{"role": "user", "content": user_message}])
        return jsonify({
            "reply": result.get("content", ""),
            "used_llm": True,
            "session_id": session_id
        })
    except Exception as e:
        return jsonify({"error": str(e), "session_id": session_id}), 500


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json(force=True) or {}
    eval_type = data.get("type")

    if eval_type != "fact_checking":
        return jsonify({"error": "Unsupported evaluation type."}), 400

    try:
        metrics = run_fact_check_evaluation(
            config_path="config",  # still points to dataset only
            dataset_path=data.get("dataset_path", "data/factchecking/sample.json"),
            num_samples=int(data.get("num_samples", 50)),
            create_negatives=bool(data.get("create_negatives", True)),
            write_outputs=bool(data.get("write_outputs", False)),
            output_dir=data.get("output_dir", "eval_outputs/factchecking"),
        )
        return jsonify({"type": eval_type, "metrics": metrics})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
