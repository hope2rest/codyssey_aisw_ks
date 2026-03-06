"""
Part C: 규칙 기반 감성 분석
"""


def rule_based_predict(text: str, sentiment_dict: dict) -> int:
    """규칙 기반 감성 예측 (부정어/강조어 처리, 1=긍정, 0=부정)"""
    positive = sentiment_dict["positive"]
    negative = sentiment_dict["negative"]
    negation = set(sentiment_dict["negation"])
    intensifier = sentiment_dict["intensifier"]

    tokens = text.split()
    total_score = 0.0
    i = 0
    while i < len(tokens):
        token = tokens[i]
        # Check if negation
        if token in negation and i + 1 < len(tokens):
            next_token = tokens[i + 1]
            score = positive.get(next_token, negative.get(next_token, 0.0))
            total_score += score * -1
            i += 2
            continue
        # Check if intensifier
        if token in intensifier and i + 1 < len(tokens):
            multiplier = intensifier[token]
            next_token = tokens[i + 1]
            score = positive.get(next_token, negative.get(next_token, 0.0))
            total_score += score * multiplier
            i += 2
            continue
        # Regular token
        score = positive.get(token, negative.get(token, 0.0))
        total_score += score
        i += 1

    return 1 if total_score > 0 else 0


def compute_metrics(predictions: list, labels: list) -> dict:
    """Accuracy, Precision, Recall, F1 계산"""
    n = len(predictions)
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    accuracy = correct / n if n > 0 else 0.0

    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }
