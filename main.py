'''from loader import prepare_dataset
from setup import load_model_and_tokenizer, generate_response
from reward import extract_number
from trainer import train

def evaluate(model, tokenizer, max_samples=20):
    dataset = prepare_dataset(split="test", max_samples=max_samples)
    correct = 0
    for ex in dataset:
        response = generate_response(model, tokenizer, ex["prompt"])
        predicted = extract_number(response)
        try:
            actual = float(ex["answer"].replace(",", ""))
        except:
            continue
        if predicted == actual:
            correct += 1
        print(f"  Predicted: {predicted} | Actual: {actual} | {'OK' if predicted==actual else 'WRONG'}")
    print(f"\nAccuracy: {correct}/{len(dataset)} = {correct/len(dataset)*100:.1f}%")

def run(test_mode=False):
    samples = 8 if test_mode else 200
    print("=== Math RL Agent ===")
    dataset = prepare_dataset(split="train", max_samples=samples)
    model, tokenizer = load_model_and_tokenizer("gpt2")
    train(model, tokenizer, dataset)
    evaluate(model, tokenizer, max_samples=4 if test_mode else 30)

import sys
if __name__ == "__main__":
    run(test_mode="--test" in sys.argv)'''


from loader import prepare_dataset
from setup import generate_response, load_model_and_tokenizer
from reward import extract_number, compute_reward
import sys


def evaluate(max_samples=10):
    dataset = prepare_dataset(split="test", max_samples=max_samples)
    correct = 0

    for ex in dataset:
        response = generate_response(ex["prompt"])
        predicted = extract_number(response)

        try:
            actual = float(ex["answer"].replace(",", ""))
        except:
            continue

        is_correct = (predicted == actual)
        if is_correct:
            correct += 1

        reward = compute_reward(response, ex["answer"])
        status = "✓" if is_correct else "✗"
        print(f"\n[{status}] Problem: {ex['prompt'].split('Problem:')[1].split('Solution:')[0].strip()[:60]}...")
        print(f"    Response : {response[:100]}...")
        print(f"    Predicted: {predicted} | Actual: {actual} | Reward: {reward}")

    print(f"\nAccuracy: {correct}/{len(dataset)} = {correct/len(dataset)*100:.1f}%")


def solve_one(question):
    """Single question solve karo — app ke liye useful"""
    prompt = (
        f"Solve this math problem step by step.\n\n"
        f"Problem: {question}\n\n"
        f"Solution:"
    )
    response = generate_response(prompt)
    answer = extract_number(response)
    print(f"\nQuestion: {question}")
    print(f"Solution:\n{response}")
    print(f"\nFinal Answer: {answer}")
    return response, answer


def run(test_mode=False):
    samples = 4 if test_mode else 20
    print("=== Math RL Agent (Groq) ===\n")
    load_model_and_tokenizer()  # sirf message print karega
    evaluate(max_samples=samples)


if __name__ == "__main__":
    # Ek custom question test karna ho toh:
    # solve_one("If a train travels 60 km/h for 2.5 hours, how far does it go?")

    run(test_mode="--test" in sys.argv)