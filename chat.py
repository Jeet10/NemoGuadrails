import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from nemoguardrails import LLMRails, RailsConfig

# Local helper for evaluation
from evaluation_utils import run_fact_check_evaluation

# --- Load environment ---
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    raise RuntimeError("NVIDIA_API_KEY environment variable not set.")

# --- Define NVIDIA LLM (main) ---
main_llm = ChatNVIDIA(
    model="deepseek-ai/deepseek-r1",
    api_key=api_key,
    temperature=0.6,
    top_p=0.7,
    max_completion_tokens=4096,
)

# --- Load NeMo Guardrails config using from_content ---
def load_rails_config():
    with open("config/rails.co", "r", encoding="utf-8") as f:
        colang_content = f.read()

    with open("config/config.yml", "r", encoding="utf-8") as f:
        yaml_content = f.read()

    # prompts.yml is optional, but if you have it include it too
    prompts_path = "config/prompts.yml"
    if os.path.exists(prompts_path):
        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts_content = f.read()
        # merge prompts.yml into yaml_content
        yaml_content = yaml_content + "\n" + prompts_content

    return RailsConfig.from_content(
        colang_content=colang_content,
        yaml_content=yaml_content,
    )

rails_config = load_rails_config()
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

    messages = [{"role": "user", "content": user_message}]

    try:
        result = rails.generate(messages=messages)
        assistant_reply = result.get("content", "")

        return jsonify({
            "reply": assistant_reply,
            "used_llm": True,
            "session_id": session_id
        })
    except Exception as e:
        return jsonify({"error": str(e), "session_id": session_id}), 500


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json(force=True) or {}
    eval_type = data.get("type")
    if not eval_type:
        return jsonify({"error": "Missing 'type' in request body."}), 400

    if eval_type not in {"fact_checking"}:
        return jsonify({"error": f"Unsupported evaluation type '{eval_type}'."}), 400

    try:
        if eval_type == "fact_checking":
            config_path = data.get("config_path", "config")
            dataset_path = data.get("dataset_path", "data/factchecking/sample.json")
            num_samples = int(data.get("num_samples", 50))
            create_negatives = bool(data.get("create_negatives", True))
            write_outputs = bool(data.get("write_outputs", False))
            output_dir = data.get("output_dir", "eval_outputs/factchecking")

            metrics = run_fact_check_evaluation(
                config_path=config_path,
                dataset_path=dataset_path,
                num_samples=num_samples,
                create_negatives=create_negatives,
                write_outputs=write_outputs,
                output_dir=output_dir,
            )

            return jsonify({"type": eval_type, "metrics": metrics})

    except FileNotFoundError as fe:
        return jsonify({"error": f"File not found: {fe}"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Unknown evaluation error."}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
