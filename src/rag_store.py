from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class GroundingRAGStore:
    """
    Retrieval-Augmented Generation (RAG) Grounding Store.
    Uses TF-IDF + Cosine Similarity over historical @AppleSupport resolution threads
    to retrieve similar past cases and ground drafted replies.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.historical_docs = []
        self.doc_vectors = None
        self.is_fitted = False

    def build_index(self, historical_pairs: List[Dict[str, Any]]):
        """
        Builds vector retrieval index from historical support threads.
        """
        self.historical_docs = historical_pairs
        if not historical_pairs:
            self.is_fitted = False
            return

        corpus = [doc.get("customer_text", "") for doc in historical_pairs]
        self.doc_vectors = self.vectorizer.fit_transform(corpus)
        self.is_fitted = True

    def retrieve_similar(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves top_k most similar historical customer inquiries and how AppleSupport resolved them.
        """
        if not self.is_fitted or self.doc_vectors is None or len(self.historical_docs) == 0:
            return []

        query_vec = self.vectorizer.transform([query_text])
        scores = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            similarity_score = float(scores[idx])
            doc = self.historical_docs[idx]
            results.append({
                "similarity_score": round(similarity_score, 4),
                "customer_text": doc.get("customer_text", ""),
                "historical_reply": doc.get("brand_reply", "")
            })
        return results

    def format_grounding_context(self, query_text: str, top_k: int = 3) -> str:
        """
        Formats retrieved historical resolution examples into prompt context text.
        """
        retrieved = self.retrieve_similar(query_text, top_k=top_k)
        if not retrieved:
            return "No prior historical resolution examples found."

        context_blocks = []
        for i, item in enumerate(retrieved, 1):
            block = (
                f"--- Example {i} (Similarity: {item['similarity_score']}) ---\n"
                f"Historical Customer Query: \"{item['customer_text']}\"\n"
                f"Historical @AppleSupport Resolution: \"{item['historical_reply']}\""
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)
