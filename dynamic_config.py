# dynamic_config.py
from nemoguardrails import RailsConfig

# === Modular Colang rules ===
COLANG_RULES = {
    "greet": """
    define user greeting
      "hi"
      "hello"
      "hey"
      "good morning"
      "good afternoon"
      "good evening"

    define bot greeting response
      "Hi,👋 I am NemoGuardrails Agent, How can I Help you today?"

    define flow greet
      user greeting
      bot greeting response
    """,

    "email": """
    define user email
      "{email:EMAIL}"

    define bot email response
      "Please Don't share PII Information"

    define flow emailResponse
      user email
      bot email response
    """,

    "toxic_language": """
    define user toxic
      "shut up"
      "stupid"
      "idiot"
      "dumb"
      "hate you"
      "f***"
      "b***"
      "kill yourself"
      "go to hell"

    define bot toxic response
      "⚠️ Please avoid using toxic or offensive language."

    define flow toxic language
      user toxic
      bot toxic response
    """,
}


def build_dynamic_config(api_key: str, enabled_rules: list[str] | None = None):
    """Build NeMo Guardrails config dynamically with only selected Colang rules."""

    # === Base config (same as before) ===
    config_dict = {
        "models": [
            {
                "type": "main",
                "engine": "nim",
                "model": "deepseek-ai/deepseek-r1",
                "parameters": {
                    "api_key": api_key,
                    "nim_base_url": "https://ai.api.nvidia.com",
                },
            },
            {
                "type": "llama_guard",
                "engine": "nim",
                "model": "deepseek-ai/deepseek-r1",
                "parameters": {
                    "api_key": api_key,
                    "nim_base_url": "https://ai.api.nvidia.com",
                },
            },
        ],
        "rails": {
            "input": {"flows": []},
            "output": {"flows": ["self check facts"]},
        },
        "lowest_temperature": 0.1,
        "prompts": [
            {
                "task": "self_check_facts",
                "content": """You are given evidence passages and a candidate answer (hypothesis).
Determine if the answer is fully grounded in, and entailed by, ONLY the evidence.
Answer strictly with "yes" or "no".
evidence: {{ evidence }}
hypothesis: {{ response }}
entails:""",
            }
        ],
    }

    # === Select Colang rules dynamically ===
    enabled_rules = enabled_rules or []
    combined_colang = "\n\n".join([COLANG_RULES[r] for r in enabled_rules if r in COLANG_RULES])

    # Build final config object
    rails_config = RailsConfig.from_content(
        config=config_dict,
        colang_content=combined_colang,
    )
    return rails_config
