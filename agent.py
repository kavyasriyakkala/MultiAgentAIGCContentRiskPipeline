"""
ReAct-style orchestrator for multi-agent fake review detection pipeline.
Coordinates Tool 1 (similarity), Tool 2 (classifier), and Tool 3 (judge).
"""
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from tools.similarity_tool import SimilarityTool
from tools.classifier_tool import ClassifierTool
from tools.judge_tool import JudgeTool

class TraceLogger:
    """Local JSON logging for execution traces"""

    def __init__(self):
        self.traces = []

    def _make_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        return obj

    def log_tool_call(self, tool_name: str, input_data: Dict, output_data: Dict, latency: float):
        """Log a tool call"""
        trace = {
            'timestamp': time.time(),
            'tool_name': tool_name,
            'input': input_data,
            'output': output_data,
            'latency_seconds': latency
        }
        self.traces.append(trace)

    def save_traces(self, output_dir='logs'):
        """Save traces to JSONL file"""
        Path(output_dir).mkdir(exist_ok=True)
        trace_file = Path(output_dir) / 'trace.jsonl'

        with open(trace_file, 'w') as f:
            for trace in self.traces:
                trace_serializable = self._make_serializable(trace)
                f.write(json.dumps(trace_serializable) + '\n')

        print(f"Traces saved to {trace_file}")

class Orchestrator:
    """
    Main orchestrator: ReAct-style decision loop.
    Coordinates three tools with decision rules.
    """

    def __init__(self):
        self.tool1 = SimilarityTool()
        self.tool2 = ClassifierTool(model_type='xgboost')
        self.tool3 = JudgeTool()
        self.logger = TraceLogger()

        self.results = []
        self.decisions_trace = []

    def train_classifier(self, train_df: pd.DataFrame):
        """Train classifier on training data"""
        print("\n=== Training Classifier ===")
        review_texts = train_df['review_text'].tolist()
        labels = train_df['label'].values

        # Embed
        embeddings, _ = self.tool1.embed_reviews(review_texts)

        # Train classifier
        self.tool2.train(review_texts, embeddings, labels)

    def process_batch(self, review_texts: List[str], review_ids: List[int] = None,
                     confidence_threshold: float = 0.65) -> Tuple[List[Dict], Dict]:
        """
        Process a batch of reviews through the pipeline.

        Decision logic:
        1. Run Tool 1 (similarity clustering)
        2. Run Tool 2 (classifier)
        3. If Tool 1 and Tool 2 agree with high confidence: finalize without judge
        4. If they disagree or confidence < threshold: call Tool 3 (judge) for tiebreaker
        5. Log reasoning for every item

        Args:
            review_texts: list of review strings
            review_ids: optional list of indices
            confidence_threshold: use judge if confidence < this value

        Returns:
            (results, stats) where:
                - results: list of dicts with final predictions
                - stats: dict with execution stats
        """
        if review_ids is None:
            review_ids = list(range(len(review_texts)))

        batch_results = []
        judge_calls = 0
        start_time = time.time()

        print(f"\nProcessing {len(review_texts)} reviews...")

        # Step 1: Run Tool 1 (similarity clustering)
        print("Step 1: Running similarity clustering tool...")
        tool1_output = self.tool1(review_texts)
        cluster_labels = tool1_output['cluster_labels']
        cluster_risk_scores = tool1_output['cluster_risk_scores']

        # Step 2: Run Tool 2 (classifier)
        print("Step 2: Running classifier tool...")
        tool2_output = self.tool2.predict(review_texts, tool1_output['embeddings'])
        predictions = tool2_output['predictions']
        probabilities = tool2_output['probabilities']
        confidence = tool2_output['confidence']

        # Step 3-5: Decision logic per review
        print("Step 3: Making tiebreaker decisions...")
        for i in tqdm(range(len(review_texts)), desc="Processing reviews"):
            review_id = review_ids[i]
            review_text = review_texts[i]

            # Tool 1 signal: interpret cluster risk score as a "fake" signal
            tool1_signal = cluster_risk_scores[i]

            # Tool 2 signal
            tool2_pred = predictions[i]
            tool2_conf = confidence[i]
            tool2_prob = probabilities[i]

            # Decision: do we need the judge?
            # Agreement = both tools predict same class
            tool1_binary = 1 if tool1_signal > 0.5 else 0
            agreement = (tool1_binary == tool2_pred)
            need_judge = not agreement or tool2_conf < confidence_threshold

            decision_reasoning = ""

            if not need_judge:
                # Confident agreement -> finalize without judge
                final_verdict = tool2_pred
                decision_reasoning = f"Tool1&2 agree (risk={tool1_signal:.2f}, conf={tool2_conf:.2f})"

                tool3_output = {
                    'verdict': -1,
                    'reasoning': 'Not called (confident agreement)',
                    'latency_seconds': 0.0,
                    'tool_name': 'judge',
                    'api_called': False
                }
            else:
                # Disagreement or low confidence -> call judge
                judge_calls += 1

                tool3_output = self.tool3(
                    review_text,
                    tool1_signal=tool1_signal,
                    tool2_prediction=tool2_pred,
                    tool2_confidence=tool2_conf
                )

                final_verdict = tool3_output['verdict']
                if final_verdict == -1:
                    # Judge couldn't decide -> use classifier
                    final_verdict = tool2_pred

                decision_reasoning = (
                    f"Tool1={tool1_binary} Tool2={tool2_pred} (conf={tool2_conf:.2f}) "
                    f"-> Judge called -> {final_verdict} ({tool3_output['reasoning'][:50]})"
                )

            # Log this decision
            result_item = {
                'review_id': review_id,
                'review_text': review_text,
                'tool1_cluster_label': int(cluster_labels[i]),
                'tool1_risk_score': float(tool1_signal),
                'tool2_prediction': int(tool2_pred),
                'tool2_probability': float(tool2_prob),
                'tool2_confidence': float(tool2_conf),
                'tool3_verdict': int(final_verdict),
                'tool3_reasoning': tool3_output.get('reasoning', ''),
                'judge_called': need_judge,
                'final_prediction': int(final_verdict),
                'decision_reasoning': decision_reasoning,
                'latency_tool1_seconds': float(tool1_output.get('latency_seconds', 0) / len(review_texts)),
                'latency_tool2_seconds': float(tool2_output.get('latency_seconds', 0) / len(review_texts)),
                'latency_tool3_seconds': float(tool3_output.get('latency_seconds', 0))
            }

            batch_results.append(result_item)
            self.results.append(result_item)
            self.decisions_trace.append({
                'review_id': review_id,
                'decision': decision_reasoning
            })

            # Log to trace
            self.logger.log_tool_call(
                f'orchestrator_review_{review_id}',
                input_data={
                    'review_text': review_text[:100],
                    'confidence_threshold': confidence_threshold
                },
                output_data={'final_prediction': final_verdict, 'judge_called': need_judge},
                latency=result_item['latency_tool1_seconds'] + result_item['latency_tool2_seconds'] + result_item['latency_tool3_seconds']
            )

        batch_latency = time.time() - start_time

        stats = {
            'total_reviews': len(review_texts),
            'judge_calls': judge_calls,
            'judge_call_rate': judge_calls / len(review_texts),
            'local_only_rate': 1 - (judge_calls / len(review_texts)),
            'batch_latency_seconds': batch_latency,
            'avg_latency_per_review': batch_latency / len(review_texts)
        }

        return batch_results, stats

    def evaluate(self, results: List[Dict], ground_truth_labels: np.ndarray) -> Dict:
        """
        Compute metrics: precision, recall, F1, AUROC.

        Args:
            results: list of result dicts from process_batch
            ground_truth_labels: np.ndarray of true labels (0=genuine, 1=fake)

        Returns:
            dict with metrics
        """
        from sklearn.metrics import (
            precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, accuracy_score
        )

        predictions = np.array([r['final_prediction'] for r in results])
        probabilities = np.array([r['tool2_probability'] for r in results])

        # Compute metrics
        accuracy = accuracy_score(ground_truth_labels, predictions)
        precision = precision_score(ground_truth_labels, predictions, zero_division=0)
        recall = recall_score(ground_truth_labels, predictions, zero_division=0)
        f1 = f1_score(ground_truth_labels, predictions, zero_division=0)

        # AUROC
        try:
            auroc = roc_auc_score(ground_truth_labels, probabilities)
        except:
            auroc = 0.0

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(ground_truth_labels, predictions).ravel()

        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auroc': float(auroc),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }

        return metrics

    def save_traces(self):
        """Save execution traces"""
        self.logger.save_traces()

    def get_results_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame"""
        return pd.DataFrame(self.results)
