import re

def extract_numbers(text):
    if text is None:
        return []
    
    text = text.replace(",", "")
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)

    return [float(num) for num in numbers] if numbers else []


def compare_numbers(pred, gold, tolerance=1e-5):
    if not pred or not gold:
        return {
            "exact_match": False,
            "within_tolerance": False,
            "pred": pred,
            "gold":gold
        }

    if isinstance(gold, str):
        gold = gold.replace(",", "")
    if isinstance(pred, str):
        pred = pred.replace(",", "")

    pred = float(pred)
    gold = float(gold)

    exact_match = pred == gold 
    within_tolerance = abs(pred - gold) <= tolerance

    return {
        "exact_match": exact_match,
        "within_tolerance": within_tolerance,
        "pred": pred,
        "gold":gold
    }


def extract_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1) if match else None
