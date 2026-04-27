from datasets import load_dataset


def prepare_dataset(split="train", max_samples=200):
    print(f"Loading GSM8K ({split}, {max_samples} samples)...")
    dataset = load_dataset("gsm8k", "main", split=split)
    dataset = dataset.select(range(min(max_samples, len(dataset))))

    def format_prompt(example):
        prompt = f"Solve this math problem step by step.\n\nProblem: {example['question']}\n\nSolution:"
        answer = example["answer"].split("####")[-1].strip()
        return {"prompt": prompt, "answer": answer}

    return dataset.map(format_prompt)