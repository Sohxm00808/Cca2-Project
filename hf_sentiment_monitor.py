"""
Name : Soham Ashok Jagtap
PRN  : 1302250127
Topic: ML Framework - Hugging Face Sentiment Monitor (multiple problem statements)

Uses a Hugging Face pipeline when `transformers` is installed (pip install transformers torch).
If it is not available, a small keyword fallback runs so the script still works offline.
"""
import csv

try:
    from transformers import pipeline
    CLASSIFIER = pipeline("sentiment-analysis",
                          model="distilbert-base-uncased-finetuned-sst-2-english")
    MODE = "HuggingFace distilbert-sst2"
except Exception:
    CLASSIFIER = None
    MODE = "Keyword fallback (transformers not installed)"

POS = {"great", "love", "excellent", "profit", "growth", "smooth", "fast", "helpful", "surge", "gain", "secure"}
NEG = {"bad", "hate", "poor", "loss", "fraud", "slow", "crash", "scam", "decline", "delay", "hacked", "fail"}


def classify(text):
    if CLASSIFIER:
        r = CLASSIFIER(text[:512])[0]
        return r["label"], round(r["score"], 3)
    words = set(text.lower().replace(".", "").replace(",", "").split())
    p, n = len(words & POS), len(words & NEG)
    return ("POSITIVE", 0.6) if p >= n else ("NEGATIVE", 0.6)


# ---- Problem statements: each has its own texts and an alert threshold ----
PROBLEMS = {
    "P1 Financial news monitoring": {
        "alert_if_negative_pct": 50,
        "texts": ["Sensex surge as banking stocks post strong profit growth.",
                  "Fintech startup hit by fraud scandal, shares crash.",
                  "RBI keeps rates unchanged, markets stay calm and stable.",
                  "Quarterly loss widens as loan defaults decline recovery."]},
    "P2 UPI / payment app reviews": {
        "alert_if_negative_pct": 40,
        "texts": ["Payments are fast and the app is smooth, love it.",
                  "Transaction failed twice and refund was a long delay.",
                  "Excellent cashback offers and secure login.",
                  "Customer care is poor, money stuck for days."]},
    "P3 Bank customer support tickets": {
        "alert_if_negative_pct": 30,
        "texts": ["Thanks, the agent was very helpful and solved my issue.",
                  "Third time complaining, nobody replies, really bad service.",
                  "Card blocked after suspected fraud, fix this slow process.",
                  "Loan approval was quick and the process was great."]},
    "P4 Crypto / social media chatter": {
        "alert_if_negative_pct": 60,
        "texts": ["Bitcoin gain continues, community feels bullish and excited.",
                  "Exchange hacked, users panic and withdraw everything.",
                  "Another scam token rug pulled investors overnight.",
                  "Regulation clarity is a great step for the industry."]},
    "P5 Campus / course feedback": {
        "alert_if_negative_pct": 35,
        "texts": ["The FinTech lectures were excellent and practical.",
                  "Timetable clashes and poor lab systems, very bad experience.",
                  "Python sessions were helpful and well paced.",
                  "Assignment deadlines are too tight."]},
}


def main():
    print(f"Mode: {MODE}\n")
    rows = []
    for name, cfg in PROBLEMS.items():
        results = [classify(t) for t in cfg["texts"]]
        neg = sum(1 for lab, _ in results if lab == "NEGATIVE")
        neg_pct = 100 * neg / len(results)
        alert = "ALERT" if neg_pct >= cfg["alert_if_negative_pct"] else "OK"
        print(f"== {name} | negative {neg_pct:.0f}% (threshold {cfg['alert_if_negative_pct']}%) -> {alert}")
        for t, (lab, sc) in zip(cfg["texts"], results):
            print(f"   [{lab:8} {sc}] {t}")
            rows.append([name, t, lab, sc, alert])
        print()
    with open("sentiment_report.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["Problem", "Text", "Label", "Score", "Status"]] + rows)
    print("Saved sentiment_report.csv")


if __name__ == "__main__":
    main()
