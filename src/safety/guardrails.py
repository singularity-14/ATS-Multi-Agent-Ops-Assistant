"""
Topic guardrails — ensures agent only responds about accelerator/physics topics.
Blocks off-topic, harmful, or out-of-scope responses.
"""
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Accelerator-related topic keywords (allowlist)
ALLOWED_TOPICS = {
    'accelerator', 'lhc', 'beam', 'beams', 'magnet', 'magnets', 'quench',
    'cryogenic', 'cryogenics', 'cryo', 'rf', 'cavity', 'cavities', 'vacuum',
    'collimator', 'collimation', 'luminosity', 'injection', 'dump', 'interlock',
    'diagnostic', 'diagnostics', 'telemetry', 'anomaly', 'anomalies', 'health',
    'subsystem', 'cern', 'proton', 'physics', 'operations', 'status', 'nominal',
    'power', 'converter', 'cooling', 'access', 'emergency', 'shutdown', 'system',
    'systems', 'sector', 'arc', 'tunnel', 'detector', 'atlas', 'cms', 'alice',
    'lhcb', 'energy', 'current', 'voltage', 'frequency', 'temperature', 'pressure',
    'orbit', 'optics', 'tune', 'chromaticity', 'emittance', 'bunch', 'fill',
    'superconducting', 'helium', 'insulation', 'quench', 'protection'
}

# Blocked patterns (harmful or clearly off-topic)
BLOCKED_PATTERNS = [
    r'\b(recipe|cooking|food|restaurant)\b',
    r'\b(politics|election|president|government|party)\b',
    r'\b(stock|crypto|bitcoin|investment|trading)\b',
    r'\b(relationship|dating|romance|love)\b',
    r'\b(violent|weapon|harm|kill|attack)\b',
    r'\b(illegal|drugs|narcotic)\b',
]

@dataclass
class GuardrailResult:
    passed: bool
    flags: list[str] = field(default_factory=list)
    reason: Optional[str] = None

class GuardrailsChecker:
    """Checks query and response against topic boundaries and safety rules."""
    
    def check(self, query: str, response: str) -> dict:
        """Run all guardrail checks. Returns dict with passed, flags."""
        flags = []
        
        # 1. Check for blocked patterns in query
        query_lower = query.lower()
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                flags.append(f'Blocked pattern detected: {pattern}')
        
        # 2. Check response doesn't contain harmful content
        response_lower = response.lower()
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, response_lower, re.IGNORECASE):
                flags.append(f'Blocked content in response: {pattern}')
        
        # 3. Topic relevance check — query should have at least one accelerator term
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        topic_overlap = query_words & ALLOWED_TOPICS
        
        # Allow short queries or escalation-type queries
        is_short = len(query.split()) <= 5
        has_question_words = any(w in query_lower for w in ['what', 'how', 'why', 'when', 'status', 'help'])
        
        if not topic_overlap and not is_short:
            flags.append('Query does not relate to accelerator operations topics')
        
        # 4. Response length sanity check
        if len(response) < 10:
            flags.append('Response too short — possible error')
        
        passed = len(flags) == 0
        logger.debug(f'Guardrails: passed={passed}, flags={flags}')
        
        return {'passed': passed, 'flags': flags}
    
    def is_topic_relevant(self, query: str) -> bool:
        """Quick check if query is topic-relevant."""
        query_lower = query.lower()
        words = set(re.findall(r'\b\w+\b', query_lower))
        return bool(words & ALLOWED_TOPICS)
