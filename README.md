# 🧮 Math RL Agent

An RL-based Math Problem Solving Agent — fine-tuned using **GRPO algorithm** (same technique used in DeepSeek-R1), deployed as a full-stack production app.

---

## 🚀 Live Demo

> App deployed on Render — [your-app.onrender.com](#)
> Fine-tuned model on HuggingFace — [SHIHICOOKIE02/qwen-math-rl-merged](https://huggingface.co/SHIHICOOKIE02/qwen-math-rl-merged)

---

## 🧠 How It Works

```
Math Problem (Input)
        ↓
Qwen2.5-Math-1.5B (Fine-tuned via GRPO)
        ↓
Step-by-step Reasoning
        ↓
Final Answer + Confidence Score
```

The model was fine-tuned using **Group Relative Policy Optimization (GRPO)** — a reinforcement learning algorithm that:
- Generates multiple outputs per problem
- Scores each output using a custom reward function
- Encourages correct reasoning, discourages wrong answers
- Uses KL penalty to prevent the model from drifting too far from the base

---

## ✨ Features

- 🔢 **Math Problem Solver** — step-by-step solutions with confidence score
- 🤖 **Concept Bot** — side panel AI assistant for math & AI concepts
- ⚡ **Dual Inference** — fine-tuned model (primary) + Groq LLaMA (fallback)
- 🎨 **Clean Dark UI** — animated math symbols, responsive design
- 🌐 **Production Deployed** — Flask API + Render hosting

---

## 🏗️ Architecture

```
Frontend (HTML/CSS/JS)
        ↓
Flask REST API (app.py)
        ↓
Inference Layer (setup.py)
    ├── Fine-tuned Qwen2.5-Math (Kaggle GPU via ngrok)
    └── Groq LLaMA-3.1 (fallback when GPU offline)
        ↓
Reward Evaluation (reward.py)
```

---

## 🔬 Training Details

| Parameter | Value |
|---|---|
| Base Model | Qwen2.5-Math-1.5B-Instruct |
| Algorithm | GRPO (Group Relative Policy Optimization) |
| Dataset | GSM8K (2000 samples) |
| Epochs | 1 |
| Generations (G) | 4 |
| KL Penalty (β) | 0.04 |
| Learning Rate | 1e-5 |
| Training Platform | Kaggle T4 GPU |
| Library | Unsloth + TRL |

### Reward Function
```python
+1.0  → Correct final answer
+0.2  → Step-by-step reasoning shown
-0.5  → No answer given
-1.0  → Wrong answer
```

---

## 📁 Project Structure

```
math-rl-agent/
├── app.py          → Flask API server
├── setup.py        → Model inference (dual: fine-tuned + Groq)
├── main.py         → Training pipeline entry point
├── loader.py       → GSM8K dataset loading
├── reward.py       → Custom GRPO reward function
├── trainer.py      → GRPO training loop
├── eval.py         → Model evaluation
├── static/
│   └── index.html  → Frontend UI with side panel bot
└── requirements.txt
```

---

## ⚙️ Setup & Run

```bash
# 1. Clone karo
git clone https://github.com/SHIHICOOKIE02/math-rl-agent
cd math-rl-agent

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# 3. Dependencies
pip install -r requirements.txt

# 4. Environment variables
set GROQ_API_KEY=your_groq_api_key
set HF_TOKEN=your_hf_token

# 5. Run
python app.py
```

Open browser: `http://localhost:5000`

---

## 🔌 API Endpoints

### POST /solve
Math problem solve karo.
```json
Request:  { "question": "If a train travels 60 km/h for 2.5 hours, how far?" }
Response: { "question": "...", "solution": "...", "answer": 150.0, "confidence": 0.85 }
```

### POST /concept
Math/AI concept explain karo.
```json
Request:  { "question": "What is GRPO?" }
Response: { "question": "...", "answer": "..." }
```

### GET /health
```json
{ "status": "ok" }
```

---

## 🧪 Inference System

```python
# Primary — Fine-tuned Qwen2.5-Math (Kaggle GPU)
try:
    response = requests.post(KAGGLE_URL, timeout=10)
    # Uses YOUR fine-tuned model!

# Fallback — Groq LLaMA-3.1
except:
    response = groq_client.chat.completions.create(...)
```

---

## 📊 Results

| Model | Accuracy |
|---|---|
| GPT-2 (baseline) | 0% |
| Groq LLaMA-3.1 | 75% |
| Fine-tuned Qwen2.5-Math | ~65-70%* |

*Trained on 2000 samples — more data = higher accuracy

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask |
| ML Training | PyTorch, TRL, Unsloth |
| RL Algorithm | GRPO |
| Dataset | GSM8K |
| Base Model | Qwen2.5-Math-1.5B-Instruct |
| Inference | Kaggle T4 GPU + Groq API |
| Deployment | Render |
| Model Hosting | HuggingFace Hub |

---

## 🔮 Future Plans

- [ ] Train on full GSM8K dataset (7000+ samples)
- [ ] Permanent GPU deployment (RunPod)
- [ ] Add history — save user's past problems
- [ ] Support LaTeX math rendering
- [ ] Multi-step problem breakdown UI

---

## 📚 References

- [DeepSeek-R1 Paper](https://arxiv.org/abs/2501.12948) — GRPO inspiration
- [TRL Library](https://github.com/huggingface/trl) — GRPO implementation
- [GSM8K Dataset](https://huggingface.co/datasets/gsm8k) — Training data
- [Qwen2.5-Math](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct) — Base model

---

## 👤 Author

**SHIHICOOKIE02** — Built with curiosity, frustration, and a lot of GPU time. 🔥