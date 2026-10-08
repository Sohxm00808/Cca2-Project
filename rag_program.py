# Name: Jaylaksh Kawathekar
# PRN : 1302250524
# -----------------------------------------------------------------------------
# ML Framework: Retrieval Augmented Generation (RAG)
#
# Pipeline:
#   1. RETRIEVE : find the most relevant documents for the question (TF-IDF + cosine similarity)
#   2. AUGMENT  : build a prompt/context from the retrieved documents
#   3. GENERATE : produce the answer from that context only
#
# Generation here is extractive (picks the best sentences from the retrieved
# documents) so the program runs offline. To use a real LLM, replace
# generate_answer() with a call to a Hugging Face / API model and pass it the
# augmented prompt built in build_prompt().
#
# Install : pip install scikit-learn
# -----------------------------------------------------------------------------

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NAME = "Jaylaksh Kawathekar"
PRN = "1302250524"

# ------------------------------ KNOWLEDGE BASE -------------------------------
DOCUMENTS = [
    {"id": "DOC1", "topic": "attendance",
     "text": "Students must maintain at least 75 percent attendance to be eligible for the end semester examination. "
             "Students with medical reasons may apply for condonation with a valid certificate."},
    {"id": "DOC2", "topic": "github",
     "text": "A GitHub repository stores project code with full version history. "
             "A branch lets a team member work on changes without affecting the main branch. "
             "A pull request asks the team to review and merge the changes from one branch into another."},
    {"id": "DOC3", "topic": "machine learning",
     "text": "Machine learning is a method where a program learns patterns from data instead of following hand written rules. "
             "Supervised learning uses labelled examples, while unsupervised learning finds structure in unlabelled data."},
    {"id": "DOC4", "topic": "rag",
     "text": "Retrieval Augmented Generation combines a search step with a language model. "
             "The system first retrieves relevant documents and then generates an answer using only that retrieved context. "
             "This reduces hallucination and keeps answers grounded in trusted sources."},
    {"id": "DOC5", "topic": "library",
     "text": "The college library is open from 8 am to 8 pm on weekdays and 9 am to 2 pm on Saturdays. "
             "Students can borrow up to four books for fourteen days. A fine of two rupees per day applies on late returns."},
    {"id": "DOC6", "topic": "assignments",
     "text": "Continuous Competency Assessment assignments must be submitted before the deadline set by the faculty. "
             "Late submissions lose marks, and plagiarised work receives zero marks."},
    {"id": "DOC7", "topic": "sentiment",
     "text": "Sentiment analysis classifies text as positive, negative or neutral. "
             "Businesses use it to monitor customer reviews and social media and to raise alerts about unhappy customers."},
]

# ---------------------------------- RULES ------------------------------------
RULES = {
    # Rule 1: how many documents to retrieve
    "TOP_K": 2,
    # Rule 2: minimum similarity needed, otherwise say "I don't know"
    "MIN_SIMILARITY": 0.10,
    # Rule 3: reject empty or too-short questions
    "MIN_QUESTION_WORDS": 2,
    # Rule 4: reject very long questions
    "MAX_QUESTION_WORDS": 40,
    # Rule 5: blocked words (simple content filter)
    "BLOCKED_WORDS": {"hack", "cheat", "password", "stupid", "idiot"},
    # Rule 6: limit the answer length (sentences)
    "MAX_ANSWER_SENTENCES": 3,
    # Rule 7: always show the sources used
    "SHOW_SOURCES": True,
    # Rule 8: boost a document when its topic word appears in the question
    "TOPIC_BOOST": 0.15,
    # Rule 9: if the top two documents score almost the same, mention that the answer may be partial
    "AMBIGUITY_GAP": 0.02,
}

STOP_WORDS_NOTE = "english"


# --------------------------- RULE CHECKS (INPUT) -----------------------------
def validate_question(question):
    """Apply the input rules. Returns (ok, message)."""
    words = re.findall(r"[A-Za-z0-9']+", question.lower())
    if len(words) < RULES["MIN_QUESTION_WORDS"]:
        return False, "Rule 3: Please ask a complete question (at least 2 words)."
    if len(words) > RULES["MAX_QUESTION_WORDS"]:
        return False, "Rule 4: Question is too long. Please keep it under 40 words."
    if RULES["BLOCKED_WORDS"] & set(words):
        return False, "Rule 5: Question contains blocked words. Please rephrase it."
    return True, ""


# -------------------------------- RETRIEVAL ----------------------------------
class Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words=STOP_WORDS_NOTE)
        self.matrix = self.vectorizer.fit_transform([d["text"] for d in documents])

    def search(self, question):
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.matrix).ravel()
        q_lower = question.lower()
        # Rule 8: topic boost
        for i, doc in enumerate(self.documents):
            if doc["topic"] in q_lower:
                scores[i] += RULES["TOPIC_BOOST"]
        order = scores.argsort()[::-1][: RULES["TOP_K"]]          # Rule 1
        return [(self.documents[i], float(scores[i])) for i in order]


# -------------------------------- AUGMENT ------------------------------------
def build_prompt(question, retrieved):
    """Create the augmented prompt that would be sent to an LLM."""
    context = "\n".join(f"[{d['id']}] {d['text']}" for d, _ in retrieved)
    return (
        "Answer using ONLY the context below. If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


# -------------------------------- GENERATE -----------------------------------
def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def generate_answer(question, retrieved):
    """Extractive generation: choose the sentences that best match the question."""
    q_words = set(re.findall(r"[a-z0-9]+", question.lower()))
    candidates = []
    for doc, _ in retrieved:
        for sentence in split_sentences(doc["text"]):
            s_words = set(re.findall(r"[a-z0-9]+", sentence.lower()))
            overlap = len(q_words & s_words)
            candidates.append((overlap, sentence))
    candidates.sort(key=lambda x: x[0], reverse=True)
    best = [s for o, s in candidates if o > 0][: RULES["MAX_ANSWER_SENTENCES"]]   # Rule 6
    return " ".join(best) if best else retrieved[0][0]["text"]


# ------------------------------- RAG PIPELINE --------------------------------
def rag_answer(retriever, question):
    ok, message = validate_question(question)
    if not ok:
        return message

    retrieved = retriever.search(question)
    best_score = retrieved[0][1]

    # Rule 2: not enough evidence in the knowledge base
    if best_score < RULES["MIN_SIMILARITY"]:
        return "Rule 2: I don't know. The knowledge base has no relevant information for this question."

    # Rule 2 (continued): drop weak matches so irrelevant documents are never used
    retrieved = [(d, s) for d, s in retrieved if s >= RULES["MIN_SIMILARITY"]]

    prompt = build_prompt(question, retrieved)       # AUGMENT (can be sent to an LLM)
    answer = generate_answer(question, retrieved)    # GENERATE

    # Rule 9: ambiguity note
    if len(retrieved) > 1 and abs(retrieved[0][1] - retrieved[1][1]) < RULES["AMBIGUITY_GAP"]:
        answer += " (Note: more than one document matched equally, so this answer may be partial.)"

    # Rule 7: show sources
    if RULES["SHOW_SOURCES"]:
        sources = ", ".join(f"{d['id']} ({s:.2f})" for d, s in retrieved)
        answer += f"\n   Sources: {sources}"
    return answer


def main():
    print(f"Name: {NAME} | PRN: {PRN}")
    print("=" * 70)
    retriever = Retriever(DOCUMENTS)

    test_questions = [
        "What attendance is required for the exam?",
        "What is a pull request in GitHub?",
        "How does RAG reduce hallucination?",
        "When is the library open on Saturday?",
        "What happens if I submit an assignment late?",
        "How do I hack the exam portal?",          # blocked word
        "Hi",                                      # too short
        "Who won the cricket world cup?",          # outside knowledge base
    ]
    for q in test_questions:
        print(f"\nQ: {q}")
        print(f"A: {rag_answer(retriever, q)}")

    print("\nAsk your own question (press Enter on an empty line to quit).")
    while True:
        try:
            q = input("> ").strip()
        except EOFError:
            break
        if not q:
            break
        print(f"A: {rag_answer(retriever, q)}")


if __name__ == "__main__":
    main()
