"""
Evaluation module: compute final metrics and generate report.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def generate_metrics_report(orchestrator, test_df, output_dir='results'):
    """
    Generate final metrics report and save to JSON.

    Args:
        orchestrator: Orchestrator instance with results
        test_df: test DataFrame with ground truth labels
        output_dir: directory to save metrics

    Returns:
        metrics dict
    """
    Path(output_dir).mkdir(exist_ok=True)

    # Get results
    results_df = orchestrator.get_results_dataframe()
    ground_truth = test_df['label'].values

    # Evaluate
    metrics = orchestrator.evaluate(orchestrator.results, ground_truth)

    # Compute cost efficiency metrics
    total_judge_calls = sum(1 for r in orchestrator.results if r['judge_called'])
    total_reviews = len(orchestrator.results)
    judge_call_rate = total_judge_calls / total_reviews if total_reviews > 0 else 0

    # Add context metrics
    metrics['total_reviews_processed'] = total_reviews
    metrics['total_judge_api_calls'] = total_judge_calls
    metrics['judge_call_rate_percent'] = judge_call_rate * 100
    metrics['local_only_rate_percent'] = (1 - judge_call_rate) * 100
    metrics['gemini_api_available'] = orchestrator.tool3.available
    metrics['timestamp'] = datetime.now().isoformat()

    # Save metrics
    metrics_path = Path(output_dir) / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to {metrics_path}")

    # Print summary
    print("\n=== FINAL METRICS ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUROC:     {metrics['auroc']:.4f}")
    print(f"\nCost Efficiency:")
    print(f"  Total reviews: {total_reviews}")
    print(f"  Judge API calls: {total_judge_calls}")
    print(f"  Judge call rate: {judge_call_rate*100:.1f}%")
    print(f"  Local-only rate: {(1-judge_call_rate)*100:.1f}% (COST EFFICIENCY WIN)")

    return metrics

def generate_full_report(orchestrator, test_df, train_df, sample_note='', output_dir='results'):
    """
    Generate comprehensive analysis report.
    """
    Path(output_dir).mkdir(exist_ok=True)

    metrics = generate_metrics_report(orchestrator, test_df, output_dir)

    # Create detailed report
    report = {
        'title': 'Multi-Agent AIGC Content Risk Pipeline - Evaluation Report',
        'generated': datetime.now().isoformat(),
        'dataset': {
            'source': 'Fake Reviews Dataset (Salminen et al., 2020)',
            'train_size': len(train_df),
            'test_size': len(test_df),
            'sample_note': sample_note,
            'label_distribution': test_df['label'].value_counts().to_dict()
        },
        'metrics': metrics,
        'findings': extract_findings(orchestrator, test_df)
    }

    report_path = Path(output_dir) / 'report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Full report saved to {report_path}")

    return report

def extract_findings(orchestrator, test_df):
    """Extract interesting findings from the results"""
    results_df = orchestrator.get_results_dataframe()

    findings = {}

    # Finding 1: Judge effectiveness
    judge_used = results_df[results_df['judge_called']]
    if len(judge_used) > 0:
        findings['judge_effectiveness'] = {
            'reviews_where_judge_was_called': len(judge_used),
            'judge_call_rate': f"{len(judge_used)/len(results_df)*100:.1f}%",
            'note': f'Judge was invoked on {len(judge_used)} reviews where local tools disagreed or confidence was low'
        }

    # Finding 2: Tool agreement
    agreement = (results_df['tool1_cluster_label'] != -1) & (
        (results_df['tool1_risk_score'] > 0.5) == (results_df['tool2_prediction'] == 1)
    )
    findings['tool_agreement'] = {
        'agreement_rate': f"{agreement.mean()*100:.1f}%",
        'disagreement_cases': int((~agreement).sum()),
        'note': 'When tools agreed, we avoided judge calls and saved on API costs'
    }

    # Finding 3: Confidence distribution
    findings['classifier_confidence'] = {
        'mean_confidence': float(results_df['tool2_confidence'].mean()),
        'min_confidence': float(results_df['tool2_confidence'].min()),
        'max_confidence': float(results_df['tool2_confidence'].max()),
        'low_confidence_threshold': 0.65,
        'reviews_below_threshold': int((results_df['tool2_confidence'] < 0.65).sum())
    }

    # Finding 4: Latency breakdown
    avg_tool1_latency = results_df['latency_tool1_seconds'].mean()
    avg_tool2_latency = results_df['latency_tool2_seconds'].mean()
    avg_tool3_latency = results_df['latency_tool3_seconds'].mean()

    findings['latency_breakdown'] = {
        'tool1_similarity_avg_ms': f"{avg_tool1_latency*1000:.2f}",
        'tool2_classifier_avg_ms': f"{avg_tool2_latency*1000:.2f}",
        'tool3_judge_avg_ms': f"{avg_tool3_latency*1000:.2f}",
        'note': 'Tool 3 is more expensive (API call) but used only on ambiguous cases'
    }

    return findings
