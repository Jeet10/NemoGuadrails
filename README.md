# 🛡️ NeMo Guardrails + NVIDIA NIM (DeepSeek-R1)

This project is a **Flask-based API** that integrates [NVIDIA NIM](https://build.nvidia.com/) models with [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) for safe and controllable AI interactions.  
It supports:

- ✅ **Rule-based safety rails** (Colang rules: greetings, PII/email detection, toxic language filtering)  
- ✅ **Fact-checking with custom prompts**  
- ✅ **Dynamic configuration** (enable/disable specific rules at runtime)  
- ✅ **Evaluation utilities** for automated fact-checking accuracy  

---

## 📂 Project Structure

```
├── chat.py                  # Flask API server
├── dynamic_config.py        # Builds NeMo Guardrails config dynamically
├── evaluation_utils.py      # Fact-check evaluation wrapper
├── requirements.txt         # Python dependencies
├── config/
│   ├── config.yml           # Base Guardrails model + rail configuration
│   ├── prompts.yml          # Task-specific prompts (e.g. fact-checking)
│   └── rails.co             # Static Colang safety rules
└── data/
    └── factchecking/
        └── sample.json      # Sample dataset for fact-check evaluation
```

---

## ⚙️ Installation

### 1. Clone repository
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate     # Linux / macOS
venv\Scripts\activate        # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set environment variables  
Create a `.env` file in the root directory:

```env
NVIDIA_API_KEY=your_nvidia_api_key_here
```

---

## 🚀 Running the API

Start the Flask server:
```bash
python chat.py
```

By default it runs at:  
👉 `http://0.0.0.0:8000`

---

## 📡 API Endpoints

### **1. Chat Endpoint**
`POST /api/chat`

#### Request
```json
{
  "message": "hi",
  "session_id": "1234",
  "enabled_rules": ["greet", "email", "toxic_language"]
}
```

#### Response
```json
{
  "reply": "Hi,👋 I am NemoGuardrails Agent, How can I Help you today?",
  "used_llm": true,
  "enabled_rules": ["greet", "email", "toxic_language"],
  "session_id": "1234"
}
```

---

### **2. Fact-Check Evaluation**
`POST /api/evaluate`

#### Request
```json
{
  "type": "fact_checking",
  "dataset_path": "data/factchecking/sample.json",
  "num_samples": 20,
  "create_negatives": true,
  "write_outputs": true,
  "output_dir": "eval_outputs/factchecking"
}
```

#### Response
```json
{
  "type": "fact_checking",
  "metrics": {
    "num_samples_requested": 20,
    "positive": {"correct": 18, "total": 20, "accuracy": 0.9, "avg_time_ms": 102.5},
    "negative": {"correct": 19, "total": 20, "accuracy": 0.95, "avg_time_ms": 110.4},
    "overall_accuracy": 0.925,
    "outputs_written": true,
    "output_dir": "eval_outputs/factchecking"
  }
}
```

---

### **3. Health Check**
`GET /health`

#### Response
```json
{"status": "ok"}
```

---

## 🛠️ Customization

### **Add or Edit Rules**
Rules are defined in **Colang** inside `dynamic_config.py` (`COLANG_RULES` dict).  
For example, a toxic language rule:

```co
define user toxic
  "stupid"
  "idiot"
  "f***"

define bot toxic response
  "⚠️ Please avoid using toxic or offensive language."

define flow toxic language
  user toxic
  bot toxic response
```

Enable at runtime by passing in `enabled_rules` via `/api/chat`.

---

### **Fact-Checking Prompts**
Defined in `config/prompts.yml`:

```yaml
prompts:
  - task: self_check_facts
    content: |-
      You are given evidence passages and a candidate answer (hypothesis).
      Determine if the answer is fully grounded in, and entailed by, ONLY the evidence.
      Answer strictly with "yes" or "no".
      evidence: {{ evidence }}
      hypothesis: {{ response }}
      entails:
```

---

## 📊 Evaluation Dataset

Sample dataset: `data/factchecking/sample.json`

```json
{
  "evidence": "The Eiffel Tower is located in Paris, France.",
  "question": "Where is the Eiffel Tower located?",
  "answer": "The Eiffel Tower is in Paris.",
  "incorrect_answer": "The Eiffel Tower is in Berlin."
}
```

You can add more fact-checking data here.

---

## ✅ Requirements

Main dependencies (`requirements.txt`):

```
flask>=3.0.0
python-dotenv>=1.0.0
langchain-nvidia-ai-endpoints>=0.2.0
nemoguardrails>=0.11.0
openai>=1.30.0
pydantic>=2.7.0
requests>=2.31.0
```

---

## 📌 Notes

- The **`main` model** is `deepseek-ai/deepseek-r1` from NVIDIA NIM.  
- Optional **`llama_guard`** model is kept as a guard model.  
- Rules can be dynamically enabled/disabled at runtime via `enabled_rules`.  
- Fact-check evaluation is programmatically wrapped in `evaluation_utils.py`.  

---

## 🏁 Next Steps

- Add more safety rails (e.g. PII, financial advice restrictions).  
- Extend fact-check dataset for stronger evaluation.  
- Deploy behind an API Gateway for production.  

---

## 📝 License
MIT License – free to use and modify.
