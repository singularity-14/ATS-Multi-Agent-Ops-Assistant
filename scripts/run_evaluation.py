#!/usr/bin/env python3
"""
CLI script to run the full evaluation harness.
Usage: python scripts/run_evaluation.py [--subset N]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluator import AgentEvaluator
from src.evaluation.reports import ReportGenerator
from src.graph.workflow import run_query

logging.basicConfig(level=logging.WARNING)

def main():
    parser = argparse.ArgumentParser(description='Run ATS Agent Evaluation Suite')
    parser.add_argument('--subset', type=int, default=0, help='Number of tests to run (0 = all)')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()
    
    # Load test suite
    test_path = Path('data/test_cases/eval_suite.json')
    if not test_path.exists():
        print(f'ERROR: Test suite not found at {test_path}')
        sys.exit(1)
    
    with open(test_path, encoding='utf-8') as f:
        test_suite = json.load(f)
    
    if args.subset > 0:
        test_suite = test_suite[:args.subset]
        print(f'Running subset of {args.subset} test cases')
    else:
        print(f'Running full suite of {len(test_suite)} test cases')
    
    # Run evaluation
    evaluator = AgentEvaluator()
    reporter = ReportGenerator()
    
    report = evaluator.run_full_eval(test_suite, run_query, verbose=args.verbose)
    
    # Print report
    reporter.print_summary(report)
    
    # Save report
    output_path = Path(args.output) if args.output else None
    saved_path = reporter.save_json(report, output_path)
    
    print(f'\nEvaluation complete. Report saved to {saved_path}')
    
    # Exit with error if pass rate is below threshold
    if report.domain_accuracy < 0.70:
        print(f'FAIL: Domain accuracy {report.domain_accuracy:.1%} below 70% threshold')
        sys.exit(1)
    
    print(f'PASS: Domain accuracy {report.domain_accuracy:.1%} meets threshold')

if __name__ == '__main__':
    main()
