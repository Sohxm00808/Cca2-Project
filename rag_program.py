"""
Name : Soham Ashok Jagtap
PRN  : 1302250127
Topic: ML Framework - Retrieval Augmented Generation (RAG) with extra rules

Pipeline: question -> rules check -> retrieve top documents (TF-IDF) -> build grounded answer + sources.
Runs fully offline. The 'generation' step is extractive (answer built only from retrieved text),
which keeps answers grounded in the knowledge base.
"""
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KNOWLEDGE_BASE = {
    "KB1-UPI": "UPI is an instant real-time payment system in India built by NPCI. Users can transfer money between bank accounts using a virtual payment address. Transactions work 24x7.",
    "KB2-CreditScore": "A credit score in India usually ranges from 300 to 900. A score above 750 is considered good. It depends on repayment history, credit utilisation and length of credit history.",
    "KB3-SIP": "A Systematic Investment Plan lets an investor put a fixed amount into a mutual fund at regular intervals. It averages the purchase cost and builds discipline.",
    "KB4-EMI": "EMI stands for equated monthly instalment. It is the fixed payment a borrower makes every month to repay a loan, covering principal and interest.",
    "KB5-KYC": "KYC means Know Your Customer. Banks verify identity and address using documents such as Aadhaar and PAN to prevent fraud and money laundering.",
    "KB6-Blockchain": "A blockchain is a distributed ledger where records are grouped into blocks linked by cryptographic hashes, making past records very hard to change.",
}

# ------------------------- RULES -------------------------
MIN_QUESTION_WORDS = 3          # R1: reject very short questions
TOP_K = 2                       # R2: use at most 2 documents
MIN_SIMILARITY = 0.12           # R3: below this -> "I don't know"
MAX_ANSWER_SENTENCES = 3        # R4: keep answers short
ADVICE_WORDS = {"buy", "sell", "invest in", "which stock", "guarantee"}   # R5: no personal investment advice
PII_PATTERN = re.compile(r"\b\d{10,16}\b|[\w.]+@[\w.]+")                  # R6: mask phone/card/account/email

vec = TfidfVectorizer(stop_words="english")
DOC_IDS = list(KNOWLEDGE_BASE)
MATRIX = vec.fit_transform(KNOWLEDGE_BASE.values())


def answer(question):
    q = PII_PATTERN.sub("[MASKED]", question.strip())                      # R6
    if len(q.split()) < MIN_QUESTION_WORDS:                                # R1
        return q, "Rule R1: question too short, please ask a complete question.", []
    if any(w in q.lower() for w in ADVICE_WORDS):                          # R5
        return q, "Rule R5: I cannot give personal investment advice. I can explain concepts only.", []

    sims = cosine_similarity(vec.transform([q]), MATRIX)[0]
    ranked = sorted(zip(DOC_IDS, sims), key=lambda x: x[1], reverse=True)[:TOP_K]   # R2
    ranked = [(d, s) for d, s in ranked if s >= MIN_SIMILARITY]            # R3
    if not ranked:
        return q, "Rule R3: no reliable source found in the knowledge base. I don't know.", []

    sentences = []
    for d, _ in ranked:
        sentences += re.split(r"(?<=[.!?])\s+", KNOWLEDGE_BASE[d])
    best = sorted(sentences, key=lambda s: cosine_similarity(vec.transform([q]), vec.transform([s]))[0][0],
                  reverse=True)[:MAX_ANSWER_SENTENCES]                     # R4
    return q, " ".join(best), [(d, round(float(s), 2)) for d, s in ranked]  # R7: always cite sources


if __name__ == "__main__":
    tests = [
        "What is UPI and how does it work?",
        "How is a credit score calculated?",
        "EMI?",
        "Which stock should I buy tomorrow?",
        "My phone is 9876543210, explain KYC documents please",
        "Who won the cricket world cup?",
    ]
    for t in tests:
        q, a, src = answer(t)
        print(f"Q: {q}\nA: {a}\nSources: {src if src else 'none'}\n")
