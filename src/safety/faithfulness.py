"""
Faithfulness checker — ensures LLM responses are grounded in retrieved context.
Uses token overlap scoring (lightweight, no extra model needed).
"""
import re
import math
import logging
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)

STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'up',
    'about', 'into', 'through', 'during', 'and', 'or', 'but', 'nor',
    'so', 'yet', 'both', 'either', 'neither', 'not', 'only', 'own',
    'same', 'than', 'too', 'very', 'just', 'as', 'if', 'this', 'that',
    'it', 'its', 'they', 'them', 'their', 'we', 'our', 'you', 'your',
    'i', 'my', 'he', 'she', 'his', 'her', 'which', 'who', 'what'
}

def tokenize(text: str) -> list[str]:
    """Lowercase, split, remove stopwords and punctuation."""
    tokens = re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

def compute_tf(tokens: list[str]) -> dict[str, float]:
    counts = Counter(tokens)
    total = len(tokens) or 1
    return {term: count / total for term, count in counts.items()}

def compute_idf(term: str, documents: list[list[str]]) -> float:
    n_docs = len(documents)
    n_containing = sum(1 for doc in documents if term in doc)
    return math.log((n_docs + 1) / (n_containing + 1)) + 1

class FaithfulnessChecker:
    """Scores how faithful a response is to retrieved context.
    
    Score 1.0 = fully grounded in context
    Score 0.0 = no overlap with context (potential hallucination)
    """
    
    def score(self, response: str, context: str) -> float:
        """Compute faithfulness score [0.0, 1.0]."""
        if not context or not response:
            return 1.0  # No context to check against
        
        response_tokens = tokenize(response)
        context_tokens = tokenize(context)
        
        if not response_tokens or not context_tokens:
            return 1.0
        
        # Method 1: Unigram overlap (Jaccard-style)
        resp_set = set(response_tokens)
        ctx_set = set(context_tokens)
        
        if not resp_set:
            return 1.0
        
        overlap = len(resp_set & ctx_set)
        precision = overlap / len(resp_set)  # How much of response is in context
        recall = overlap / len(ctx_set) if ctx_set else 0  # How much context is covered
        
        # F1-like combination but weighted toward precision
        # High precision = response uses context terms
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        
        # Weighted: precision matters more for faithfulness
        faith_score = 0.7 * precision + 0.3 * f1
        
        # Clamp to [0, 1]
        faith_score = max(0.0, min(1.0, faith_score))
        
        logger.debug(f'Faithfulness: precision={precision:.2f}, recall={recall:.2f}, score={faith_score:.2f}')
        return round(faith_score, 3)
    
    def is_faithful(self, response: str, context: str, threshold: float = 0.3) -> bool:
        """Returns True if response is sufficiently grounded in context."""
        return self.score(response, context) >= threshold
    
    def explain(self, response: str, context: str) -> dict:
        """Detailed faithfulness analysis with explanation."""
        score = self.score(response, context)
        response_tokens = set(tokenize(response))
        context_tokens = set(tokenize(context))
        
        grounded_terms = list(response_tokens & context_tokens)[:20]
        ungrounded_terms = list(response_tokens - context_tokens)[:20]
        
        return {
            'score': score,
            'is_faithful': score >= 0.3,
            'grounded_terms': grounded_terms,
            'potentially_hallucinated_terms': ungrounded_terms,
            'assessment': 'HIGH' if score >= 0.7 else ('MEDIUM' if score >= 0.3 else 'LOW')
        }
