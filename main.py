"""
Main pipeline script: orchestrates the full multi-agent AIGC detection pipeline.
"""
import os
import sys
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from data_loader import load_dataset, split_data
from agent import Orchestrator
from eval import generate_full_report

def main(args):
    """
    Run the complete pipeline:
    1. Load dataset
    2. Split train/test
    3. Train classifier
    4. Process test set through orchestrator
    5. Evaluate and generate report
    """

    print("=" * 60)
    print("Multi-Agent AIGC Content Risk Pipeline")
    print("=" * 60)

    # Step 1: Load dataset
    print("\n[1/5] Loading dataset...")
    df, sample_note = load_dataset(
        force_resample=args.force_resample,
        sample_size=args.sample_size
    )

    # Step 2: Split data
    print("\n[2/5] Splitting data (80/20)...")
    train_df, test_df = split_data(df, train_size=0.8)

    # Step 3: Initialize and train orchestrator
    print("\n[3/5] Initializing orchestrator and training classifier...")
    orchestrator = Orchestrator()
    orchestrator.train_classifier(train_df)

    # Step 4: Process test set
    print("\n[4/5] Processing test set through pipeline...")
    batch_results, stats = orchestrator.process_batch(
        review_texts=test_df['review_text'].tolist(),
        review_ids=test_df.index.tolist(),
        confidence_threshold=args.confidence_threshold
    )

    print(f"\nPipeline execution stats:")
    print(f"  Total reviews: {stats['total_reviews']}")
    print(f"  Judge calls: {stats['judge_calls']}")
    print(f"  Judge call rate: {stats['judge_call_rate']*100:.1f}%")
    print(f"  Batch latency: {stats['batch_latency_seconds']:.2f}s")
    print(f"  Avg per review: {stats['avg_latency_per_review']:.4f}s")

    # Step 5: Evaluate and generate report
    print("\n[5/5] Generating evaluation report...")
    report = generate_full_report(
        orchestrator,
        test_df,
        train_df,
        sample_note=sample_note,
        output_dir=args.output_dir
    )

    # Save traces
    orchestrator.save_traces()

    # Save detailed results
    results_df = orchestrator.get_results_dataframe()
    results_csv = Path(args.output_dir) / 'detailed_results.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"Detailed results saved to {results_csv}")

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print(f"Results saved to {args.output_dir}/")
    print("=" * 60)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Multi-Agent AIGC Content Risk Pipeline'
    )
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Directory to save results (default: results/)'
    )
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.65,
        help='Confidence threshold for judge calls (default: 0.65)'
    )
    parser.add_argument(
        '--force-resample',
        action='store_true',
        help='Force resample dataset to sample size'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=5000,
        help='Target sample size if resampling (default: 5000)'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    try:
        main(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
