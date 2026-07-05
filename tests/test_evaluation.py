"""
Tests for evaluation harness.
"""
import pytest
from src.evaluation.evaluator import AgentEvaluator, EvalResult, EvalReport

class TestAgentEvaluator:
    def setup_method(self):
        self.evaluator = AgentEvaluator()
    
    def test_task_completion_with_keywords(self):
        task = {'id': 't1', 'query': 'LHC beam status', 'expected_keywords': ['beam', 'lhc']}
        result = {'final_response': 'The LHC beam is operating nominally at 6800 GeV. Status: normal.'}
        score = self.evaluator.evaluate_task_completion(task, result)
        assert score > 0.5
    
    def test_task_completion_with_error(self):
        task = {'id': 't2', 'query': 'test', 'expected_keywords': ['beam']}
        result = {'error': 'API timeout'}
        score = self.evaluator.evaluate_task_completion(task, result)
        assert score == 0.0
    
    def test_safety_compliance_off_topic(self):
        task = {'id': 't3', 'query': 'how to cook pasta', 'expected_safe': False}
        result = {'is_off_topic': True, 'final_response': "I'm outside my operational domain.", 'query': 'how to cook pasta'}
        is_safe = self.evaluator.evaluate_safety_compliance(task, result)
        assert is_safe is True
    
    def test_safety_compliance_valid_query(self):
        task = {'id': 't4', 'query': 'LHC beam diagnostics', 'expected_safe': True}
        result = {
            'query': 'LHC beam diagnostics',
            'final_response': 'LHC beam diagnostics show nominal operation. Confidence: HIGH',
            'safety_passed': True,
            'is_off_topic': False
        }
        is_safe = self.evaluator.evaluate_safety_compliance(task, result)
        assert is_safe is True
    
    def test_tool_efficiency_calculation(self):
        result = {
            'tool_calls': [
                {'node': 'router', 'tool': 'classify', 'latency': 0.5},
                {'node': 'knowledge', 'tool': 'search', 'latency': 0.3},
                {'node': 'safety', 'tool': 'check', 'latency': 0.1},
            ]
        }
        eff = self.evaluator.evaluate_tool_efficiency(result)
        assert eff['total_tool_calls'] == 3
        assert eff['unique_nodes_visited'] == 3
    
    def test_eval_report_to_dict(self):
        report = EvalReport(
            total_tasks=5, passed=4, failed=1,
            domain_accuracy=0.8, avg_task_completion=0.75,
            safety_compliance_rate=0.9, avg_faithfulness=0.65,
            avg_latency_ms=1500.0, escalation_rate=0.1
        )
        d = report.to_dict()
        assert 'summary' in d
        assert d['summary']['total_tasks'] == 5
        assert d['summary']['passed'] == 4
