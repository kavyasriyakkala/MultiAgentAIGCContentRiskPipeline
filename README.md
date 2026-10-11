# Multi-Agent AIGC Content Risk Pipeline

A cost-efficient multi-agent system for detecting AI-generated fake product reviews. Combines a local ML classifier + similarity clustering with selective LLM-as-judge fallback to achieve 85.8% accuracy while minimizing API costs.

## Overview

**Problem:** Distinguish human-written reviews from AI-generated ones in e-commerce data.

**Solution:** ReAct-style orchestrator with three specialized tools:
1. **Similarity Clustering** (local embeddings + DBSCAN) — detects templated/bot-like clusters
2. **XGBoost Classifier** (embeddings + handcrafted features) — lightweight, interpretable predictions
3. **LLM Judge** (Google Gemini, selective) — tiebreaker for uncertain cases

**Key Result:** 85.7% accuracy, 86.4% recall, 94.5% AUROC — beating a TF-IDF + Logistic Regression baseline by ~8 points. 54.2% of reviews resolved locally with no API calls; 45.8% escalated to the judge on low-confidence or disagreement.

## Dataset

**Source:** [Fake Reviews Dataset](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset) - Salminen et al. (2020)
- ~40,000 product reviews (OR = genuine, CG = AI-generated)
- Real e-commerce data with ground truth labels

## Architecture

```
For each review:
  1. Embed text → Detect clusters (Tool 1)
  2. Extract features → Classifier prediction (Tool 2)
  3. Decision: If confident agreement → Finalize
              Else → Call Judge (Tool 3) for tiebreaker
  4. Log full reasoning trace
```

**Tool Details:**

| Tool | Input | Output | Cost |
|------|-------|--------|------|
| **Similarity** | Review text | Cluster label, risk score (0-1) | None (local TF-IDF + DBSCAN) |
| **Classifier** | Embeddings + 9 features | Binary prediction, confidence | None (local XGBoost) |
| **Judge** | Review + tool outputs | Verdict + reasoning | ~$0 (free tier, 40% of reviews skip it) |

## Results

Tested on 1,000 reviews (80/20 train/test split, stratified by label):

```
Pipeline (Tool 1 + Tool 2 cascade):
  Accuracy:     85.7%
  Precision:    85.2%
  Recall:       86.4%
  F1 Score:     85.8%
  AUROC:        94.5%

Baseline (TF-IDF + Logistic Regression):
  Accuracy:     77.9%
  F1 Score:     77.9%
  AUROC:        85.3%

Improvement over baseline: +7.8pp accuracy, +9.2pp AUROC

Judge Routing:
  - Local-only decisions:    54.2% (Tools 1+2 agreed with high confidence)
  - Escalated to judge:      45.8% (disagreement or confidence < 0.65)
  - Estimated cost (judge):  ~$0 on free tier for low-volume use
```

## Installation

```bash
# Clone/download repo
cd MultiAgentAIGCContentRiskPipeline

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start (Local Only)
```bash
python main.py
```
Runs the full pipeline without Gemini API. Tools 1 & 2 resolve ~40% of cases.

### With Gemini API (Optional)
```bash
export GEMINI_API_KEY="your-free-tier-key"
python main.py
```
Enables selective LLM judge for ambiguous cases. Free tier API key from https://aistudio.google.com.

### Custom Configuration
```bash
python main.py \
  --output-dir results \
  --confidence-threshold 0.65 \
  --sample-size 5000
```

## Project Structure

```
├── main.py                 # Entry point
├── agent.py               # Orchestrator + decision logic
├── data_loader.py         # Dataset loading with fallbacks
├── eval.py                # Evaluation metrics
│
├── tools/
│   ├── similarity_tool.py  # TF-IDF embedding + DBSCAN clustering
│   ├── classifier_tool.py  # XGBoost on embeddings + handcrafted features
│   └── judge_tool.py       # Gemini API wrapper
│
├── requirements.txt
└── results/
    ├── metrics.json       # Final metrics
    ├── report.json        # Detailed analysis
    ├── detailed_results.csv # Per-review predictions
    └── logs/trace.jsonl   # Full execution traces
```

## Design Highlights

### 1. Cost-Aware Decision Logic
Only call the expensive judge (Gemini API) when:
- Tools 1 & 2 disagree, **OR**
- Tool 2 confidence < 0.65 threshold

Result: 40.7% of reviews skip API entirely, saving ~60% of potential costs.

### 2. Local-First ML
- TF-IDF embeddings (no OpenAI dependency)
- XGBoost classifier (fast, interpretable, no cloud dependency)
- Handcrafted features: length, punctuation, uniqueness, repetition patterns

Enables reproducible, deterministic inference without external APIs.

### 3. Graceful Degradation
- No Gemini key? → Tool 3 skipped, Tool 1+2 decisions still valid
- No Kaggle access? → Falls back to OSF, HuggingFace, synthetic data
- Missing features? → Classifier still trains on available data

### 4. Full Observability
- JSON trace logs for every decision + tool call
- Per-review predictions + confidence scores
- Latency breakdown by tool

## Methodology

**Training (80% of data):**
1. Embed all reviews with TF-IDF vectorizer
2. Extract 9 handcrafted features per review
3. Combine embeddings + features
4. Train XGBoost classifier

**Inference (20% of data, per review):**
1. Tool 1: Embed → DBSCAN cluster → risk_score
2. Tool 2: Feature extraction → XGBoost prediction + confidence
3. Decision logic:
   - If `tool1_signal ≈ tool2_pred` AND `confidence ≥ 0.65` → finalize
   - Else → Call Judge, use verdict
4. Log: review, all tool outputs, reasoning, final decision

**Evaluation:**
- Standard metrics: accuracy, precision, recall, F1, AUROC
- Cost efficiency: judge call rate
- Latency per tool

## Key Findings

**Finding 1: XGBoost cascade substantially outperforms baseline**
The pipeline achieves 85.7% accuracy vs. 77.9% for TF-IDF + Logistic Regression — an 8-point gap on the same data. The KNN similarity signal and handcrafted features (punctuation density, unique word ratio, repetition) add meaningful signal beyond raw TF-IDF.

**Finding 2: Tool 1 provides a real orthogonal signal**
After fixing Tool 1 to use KNN similarity to known-fake training examples (replacing broken DBSCAN), tools agree on ~54% of reviews. The 45.8% escalation rate reflects genuine uncertainty, not noise.

**Finding 3: Feature Importance**
Punctuation density, unique word ratio, and repetition patterns are strong indicators of AI-generated text, matching known characteristics of modern LLM outputs.

## Limitations & Future Work

- **Embedding Model:** TF-IDF is general-purpose; domain-specific review embeddings would improve clustering.
- **Dataset Age:** Training data is GPT-2 era; performance on GPT-4/Claude-generated reviews unknown.
- **Judge as Tiebreaker:** LLM judge doesn't significantly outperform local tools on most cases; main value is on truly ambiguous reviews (rare in test set).

## For Employers / Portfolio

**Highlight:**
> Engineered a cost-efficient multi-agent detection system achieving **85.7% accuracy / 94.5% AUROC** on 1,000 reviews — **+7.8pp over a TF-IDF + Logistic Regression baseline**. 54.2% of reviews resolved locally with no API calls; ambiguous cases escalated to a blind LLM judge. Demonstrates: multi-agent orchestration, cost-aware routing, local ML inference, LLM API integration with graceful degradation, and JSON observability tracing.

**Key Skills Demonstrated:**
- Multi-agent ReAct-style orchestration
- ML pipeline design (feature engineering, classifier training)
- API integration with fallback mechanisms
- JSON-based observability and tracing
- Kaggle data + multiple fallback sources
- Error handling and graceful degradation

## References

- **Dataset:** [Salminen et al. (2020) - Fake Reviews Dataset](https://osf.io/mryx5/)
- **XGBoost:** https://xgboost.readthedocs.io/
- **Gemini API:** https://ai.google.dev/
