"""
Report generator for evaluation results — produces rich terminal output and JSON reports.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text

from src.evaluation.evaluator import EvalReport

logger = logging.getLogger(__name__)
console = Console()

class ReportGenerator:
    """Generate and save evaluation reports."""
    
    def print_summary(self, report: EvalReport):
        """Print a rich terminal summary of evaluation results."""
        # Header
        console.print(Panel.fit(
            '[bold cyan]ATS Agent Evaluation Report[/bold cyan]',
            border_style='cyan'
        ))
        
        # Summary metrics
        table = Table(box=box.ROUNDED, show_header=True, header_style='bold magenta')
        table.add_column('Metric', style='cyan')
        table.add_column('Value', justify='right')
        table.add_column('Status', justify='center')
        
        def status_icon(val, threshold):
            return '✅' if val >= threshold else '❌'
        
        table.add_row('Total Tasks', str(report.total_tasks), '')
        table.add_row('Passed', str(report.passed), status_icon(report.passed/report.total_tasks, 0.8))
        table.add_row('Failed', str(report.failed), '')
        table.add_row('Domain Accuracy', f'{report.domain_accuracy:.1%}', status_icon(report.domain_accuracy, 0.85))
        table.add_row('Task Completion', f'{report.avg_task_completion:.1%}', status_icon(report.avg_task_completion, 0.70))
        table.add_row('Safety Compliance', f'{report.safety_compliance_rate:.1%}', status_icon(report.safety_compliance_rate, 0.95))
        table.add_row('Avg Faithfulness', f'{report.avg_faithfulness:.1%}', status_icon(report.avg_faithfulness, 0.60))
        table.add_row('Avg Latency', f'{report.avg_latency_ms:.0f}ms', '')
        table.add_row('Escalation Rate', f'{report.escalation_rate:.1%}', '')
        
        console.print(table)
        
        # Per-result table
        detail_table = Table(box=box.SIMPLE, show_header=True, header_style='bold blue', title='Per-Task Results')
        detail_table.add_column('ID', style='dim')
        detail_table.add_column('Query', max_width=40)
        detail_table.add_column('Domain', justify='center')
        detail_table.add_column('Completion', justify='right')
        detail_table.add_column('Safe', justify='center')
        detail_table.add_column('Faithful', justify='right')
        
        for r in report.results:
            domain_match = '✅' if r.domain_correct else f'❌({r.actual_domain})'
            detail_table.add_row(
                r.task_id,
                r.query[:40] + ('...' if len(r.query) > 40 else ''),
                domain_match,
                f'{r.task_completion_score:.0%}',
                '✅' if r.safety_compliant else '❌',
                f'{r.faithfulness_score:.0%}',
            )
        
        console.print(detail_table)
    
    def save_json(self, report: EvalReport, output_path: Optional[Path] = None) -> Path:
        """Save evaluation report as JSON."""
        if output_path is None:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            output_path = Path(f'data/eval_reports/eval_{timestamp}.json')
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = report.to_dict()
        data['generated_at'] = datetime.utcnow().isoformat() + 'Z'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f'Evaluation report saved to {output_path}')
        console.print(f'[green]Report saved to {output_path}[/green]')
        return output_path
