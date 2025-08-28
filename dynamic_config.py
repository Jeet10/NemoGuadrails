# dynamic_config.py
from nemoguardrails import RailsConfig


def build_dynamic_config(api_key: str):
    """Build combined NeMo Guardrails config dynamically."""

    # === Base config (was config.yml) ===
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

    # === Colang rules (was rails.co) ===
    colang_rules = """
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

    define user email
      "{email:EMAIL}"

    define bot email response
      "Please Don't share PII Information"

    define flow emailResponse
      user email
      bot email response
    """

    # Build final config object
    rails_config = RailsConfig.from_content(
        config=config_dict,
        colang_content=colang_rules,
    )
    return rails_config
