from flask import Flask, request, jsonify, send_from_directory
from setup import generate_response
from reward import extract_number, compute_reward
import os

app = Flask(__name__, static_folder="static")


# ── serve frontend ──────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ── health check ────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ── main solve endpoint ─────────────────────────────────────────
@app.route("/solve", methods=["POST"])
def solve():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "question field required"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "question cannot be empty"}), 400

    prompt = (
        "Solve this math problem step by step. "
        "Show your work clearly and state the final answer at the end.\n\n"
        f"Problem: {question}\n\n"
        "Solution:"
    )

    solution = generate_response(prompt, max_tokens=400)
    answer = extract_number(solution)
    reward = compute_reward(solution, str(answer) if answer else "")

    return jsonify({
        "question": question,
        "solution": solution,
        "answer": answer,
        "confidence": round(max(0, (reward + 1) / 2), 2),  # 0 to 1 scale
    })

@app.route("/concept", methods=["POST"])
def concept():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "question required"}), 400

    question = data["question"].strip()

    prompt = (
        "You are a helpful math and AI concept explainer. "
        "Explain concepts clearly, simply, and with examples. "
        "Keep answers concise — 3 to 5 sentences max.\n\n"
        f"Question: {question}\n\nAnswer:"
    )

    answer = generate_response(prompt, max_tokens=200)
    return jsonify({"question": question, "answer": answer})

if __name__ == "__main__":
    print("\n=== Math RL Agent API ===")
    print("Open browser: http://localhost:5000")
    print("========================\n")
    app.run(debug=True, port=5000)