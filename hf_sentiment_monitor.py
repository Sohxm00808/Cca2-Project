# Name: Jaylaksh Kawathekar
# PRN : 1302250524
# -----------------------------------------------------------------------------
# ML Framework: Hugging Face Sentiment Monitor
#
# A sentiment monitor that classifies text as POSITIVE / NEGATIVE / NEUTRAL and
# raises ALERTS when negative sentiment is strong. It is applied to several
# problem statements instead of just one.
#
# Install : pip install transformers torch
# Model   : distilbert-base-uncased-finetuned-sst-2-english (downloaded on first run)
# If the transformers library is not installed, a simple keyword-based fallback
# is used so the program still runs.
# -----------------------------------------------------------------------------

from collections import Counter

NAME = "Jaylaksh Kawathekar"
PRN = "1302250524"

# Confidence below this value is reported as NEUTRAL
NEUTRAL_THRESHOLD = 0.70
# A NEGATIVE result with confidence at or above this value raises an alert
ALERT_THRESHOLD = 0.90

# ------------------------------ PROBLEM STATEMENTS ---------------------------
PROBLEM_STATEMENTS = {
    "1. Product Reviews (E-commerce)": [
        "The headphones have amazing sound quality and the battery lasts all week.",
        "The charger stopped working after two days. Complete waste of money.",
        "Delivery was on time and the packaging was fine.",
    ],
    "2. Customer Support Tickets": [
        "Your support team solved my issue in five minutes. Brilliant service!",
        "I have been waiting for a refund for three weeks and nobody replies.",
        "I want to know the status of my order number 4521.",
    ],
    "3. Social Media Brand Monitoring": [
        "Loving the new update, the app feels so much faster now!",
        "This app keeps crashing and the latest update made it worse.",
        "Just downloaded the app to try it out.",
    ],
    "4. Student Feedback on Courses": [
        "The professor explained machine learning with great real-life examples.",
        "The lectures were boring and the assignments were confusing.",
        "The course has twelve lectures and two assignments.",
    ],
    "5. News Headlines Tone Analysis": [
        "Local startup wins national award for innovation in clean energy.",
        "Factory fire leaves hundreds without jobs as losses mount.",
        "City council to discuss the annual budget on Monday.",
    ],
    "6. Restaurant and Food Delivery Reviews": [
        "The biryani was delicious and still hot when it arrived.",
        "Food arrived cold and the order was missing two items.",
        "I ordered a vegetable sandwich and a cold coffee.",
    ],
}


# ------------------------------ MODEL LOADING --------------------------------
def load_classifier():
    """Load the Hugging Face pipeline, or fall back to a keyword classifier."""
    try:
        from transformers import pipeline

        clf = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
        print("[INFO] Using Hugging Face model: distilbert-base-uncased-finetuned-sst-2-english")
        return clf
    except Exception as err:  # library missing, no internet, etc.
        print(f"[WARN] Hugging Face model unavailable ({type(err).__name__}). Using keyword fallback.")
        return keyword_classifier


def keyword_classifier(texts):
    """Very simple fallback so the program runs without transformers."""
    positive = {"amazing", "brilliant", "loving", "great", "delicious", "faster", "wins", "solved", "award", "hot"}
    negative = {"waste", "stopped", "waiting", "crashing", "worse", "boring", "confusing", "fire", "losses", "missing"}
    results = []
    for t in texts:
        words = {w.strip(".,!?").lower() for w in t.split()}
        pos, neg = len(words & positive), len(words & negative)
        if pos > neg:
            results.append({"label": "POSITIVE", "score": 0.95})
        elif neg > pos:
            results.append({"label": "NEGATIVE", "score": 0.95})
        else:
            results.append({"label": "POSITIVE", "score": 0.55})
    return results


# ------------------------------ MONITOR LOGIC --------------------------------
def interpret(result):
    """Convert raw model output into POSITIVE / NEGATIVE / NEUTRAL and an alert flag."""
    label, score = result["label"].upper(), result["score"]
    if score < NEUTRAL_THRESHOLD:
        label = "NEUTRAL"
    alert = label == "NEGATIVE" and score >= ALERT_THRESHOLD
    return label, score, alert


def monitor(classifier, name, texts):
    print(f"\n=== {name} ===")
    results = classifier(texts)
    counts = Counter()
    alerts = 0
    for text, res in zip(texts, results):
        label, score, alert = interpret(res)
        counts[label] += 1
        alerts += int(alert)
        flag = "  <-- ALERT: strongly negative" if alert else ""
        print(f"[{label:8}] ({score:.2f}) {text}{flag}")
    total = len(texts)
    health = (counts["POSITIVE"] - counts["NEGATIVE"]) / total * 100
    print(f"Summary: {dict(counts)} | Alerts: {alerts} | Sentiment score: {health:+.0f}")
    return counts, alerts


def main():
    print(f"Name: {NAME} | PRN: {PRN}")
    print("=" * 70)
    classifier = load_classifier()

    grand = Counter()
    total_alerts = 0
    for name, texts in PROBLEM_STATEMENTS.items():
        counts, alerts = monitor(classifier, name, texts)
        grand.update(counts)
        total_alerts += alerts

    print("\n" + "=" * 70)
    print("OVERALL REPORT")
    print(f"Total texts analysed : {sum(grand.values())}")
    print(f"Distribution         : {dict(grand)}")
    print(f"Total alerts raised  : {total_alerts}")

    # Interactive mode: type your own text
    print("\nType your own sentence to analyse (press Enter on an empty line to quit).")
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if not text:
            break
        label, score, alert = interpret(classifier([text])[0])
        print(f"   {label} ({score:.2f})" + ("  <-- ALERT" if alert else ""))


if __name__ == "__main__":
    main()
