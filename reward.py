import re


def extract_number(text):
    text = text.replace(",", "")
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1])
        except:
            return None
    return None


def compute_reward(completion, correct_answer):
    reward = 0.0
    predicted = extract_number(completion)
    try:
        correct = float(str(correct_answer).replace(",", ""))
    except:
        return -1.0
    if predicted is None:
        reward -= 0.5
    elif predicted == correct:
        reward += 1.0
    else:
        reward -= 1.0
    if len([l for l in completion.split("\n") if l.strip()]) >= 2:
        reward += 0.2
    return round(reward, 3)


def batch_reward_fn(completions, answer, **kwargs):
    return [compute_reward(c, a) for c, a in zip(completions, answer)]