"""
Tests for safety module: guardrails, faithfulness checker, audit logger.
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.safety.guardrails import GuardrailsChecker
from src.safety.faithfulness import FaithfulnessChecker, tokenize
from src.safety.audit_logger import AuditLogger

# --- Guardrails Tests ---

class TestGuardrailsChecker:
    def setup_method(self):
        self.checker = GuardrailsChecker()
    
    def test_valid_accelerator_query_passes(self):
        result = self.checker.check(
            'What is the LHC beam injection procedure?',
            'The LHC beam injection procedure involves... Confidence: HIGH'
        )
        assert result['passed'] is True
        assert len(result['flags']) == 0
    
    def test_off_topic_cooking_query_flagged(self):
        result = self.checker.check(
            'What is the recipe for pasta?',
            'To make pasta, follow this recipe...'
        )
        assert result['passed'] is False
    
    def test_blocked_political_content(self):
        result = self.checker.check(
            'Who won the election?',
            'The election results show...'
        )
        assert result['passed'] is False
    
    def test_diagnostics_query_passes(self):
        result = self.checker.check(
            'Run anomaly check on LHC_BEAM1',
            'Anomaly analysis complete. System nominal. Confidence: HIGH'
        )
        assert result['passed'] is True
    
    def test_cryogenics_query_passes(self):
        result = self.checker.check(
            'What is the cryogenics temperature for superconducting magnets?',
            'Cryogenic systems maintain 1.9K for superconducting operation. Confidence: HIGH'
        )
        assert result['passed'] is True
    
    def test_very_short_response_flagged(self):
        result = self.checker.check(
            'What is the LHC beam status?',
            'OK'
        )
        assert result['passed'] is False
    
    def test_topic_relevance_check(self):
        assert self.checker.is_topic_relevant('LHC beam dump procedure') is True
        assert self.checker.is_topic_relevant('best pizza recipe tonight') is False

# --- Faithfulness Tests ---

class TestFaithfulnessChecker:
    def setup_method(self):
        self.checker = FaithfulnessChecker()
    
    def test_high_overlap_scores_high(self):
        context = 'The LHC beam injection uses superconducting magnets at 1.9K to guide protons through the tunnel.'
        response = 'The LHC beam injection procedure uses superconducting magnets cooled to 1.9K. Confidence: HIGH'
        score = self.checker.score(response, context)
        assert score > 0.4
    
    def test_no_overlap_scores_low(self):
        context = 'The LHC uses superconducting dipole magnets cooled to 1.9K.'
        response = 'Cooking pasta requires boiling water with salt at 100 degrees.'
        score = self.checker.score(response, context)
        assert score < 0.2
    
    def test_empty_context_returns_one(self):
        score = self.checker.score('Some response', '')
        assert score == 1.0
    
    def test_empty_response_returns_one(self):
        score = self.checker.score('', 'Some context')
        assert score == 1.0
    
    def test_is_faithful_threshold(self):
        context = 'Beam current is 0.582A. Nominal energy is 6800 GeV.'
        grounded = 'The beam current is 0.582A and energy is 6800 GeV. Confidence: HIGH'
        ungrounded = 'The stock market crashed due to political events.'
        assert self.checker.is_faithful(grounded, context, threshold=0.2) is True
        assert self.checker.is_faithful(ungrounded, context, threshold=0.5) is False
    
    def test_explain_returns_dict(self):
        result = self.checker.explain('LHC beam current nominal', 'beam current is nominal at 0.582A')
        assert 'score' in result
        assert 'grounded_terms' in result
        assert 'assessment' in result
    
    def test_tokenize_removes_stopwords(self):
        tokens = tokenize('the LHC is a very large accelerator')
        assert 'the' not in tokens
        assert 'is' not in tokens
        assert 'lhc' in tokens
        assert 'accelerator' in tokens

# --- Audit Logger Tests ---

class TestAuditLogger:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = Path(self.tmp.name)
        with patch('src.safety.audit_logger.get_settings') as mock_settings:
            mock_settings.return_value.audit_db_path = self.db_path
            self.logger = AuditLogger(db_path=self.db_path)
    
    def test_log_tool_call(self):
        record_id = self.logger.log_tool_call(
            node='router',
            tool='classify_query',
            input_data={'query': 'test'},
            output_data={'domain': 'docs'},
            latency=0.5
        )
        assert record_id is not None
    
    def test_log_response(self):
        record_id = self.logger.log_response(
            query='test query',
            domain='docs',
            response='test response',
            confidence=0.9,
            safety_passed=True
        )
        assert record_id is not None
    
    def test_get_recent_tool_calls(self):
        self.logger.log_tool_call('test_node', 'test_tool', {}, {}, 0.1)
        calls = self.logger.get_recent_tool_calls(10)
        assert len(calls) >= 1
        assert calls[0]['node'] == 'test_node'
    
    def test_get_stats(self):
        self.logger.log_tool_call('n1', 't1', {}, {}, 0.2)
        self.logger.log_response('q', 'd', 'r', 0.8, True)
        stats = self.logger.get_stats()
        assert stats['total_tool_calls'] >= 1
        assert stats['total_responses'] >= 1
        assert 'avg_confidence' in stats
    
    def test_new_session(self):
        session_id = self.logger.new_session()
        assert len(session_id) == 36  # UUID format
