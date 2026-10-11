# Multi-Agent AIGC Content Risk Pipeline

A multi-agent system for detecting AI-generated fake product reviews. Uses a local ML cascade (KNN similarity routing + XGBoost classifier) with an optional LLM-as-judge fallback for ambiguous cases.

## Overview

**Problem:** Distinguish human-written reviews from AI-generated ones in e-commerce data.

**Solution:** ReAct-style orchestrator with three specialized tools:
1. **KNN Similarity** (local TF-IDF + nearest-neighbor) — scores each review by proximity to known fakes in training data; used for routing decisions
2. **XGBoost Classifier** (TF-IDF embeddings + 9 handcrafted features) — primary prediction engine
3. **LLM Judge** (Google Gemini, selective) — tiebreaker for low-confidence or disagreement cases; blind to classifier output to prevent anchoring

**Key Result:** XGBoost classifier achieves **85.7% accuracy / 94.5% AUROC** on 1,000 reviews, beating a TF-IDF + Logistic Regression baseline by **+7.8pp accuracy, +9.2pp AUROC**. The cascade routes 45.8% of reviews to the LLM judge; judge evaluation pending full API quota.

## Dataset

**Source:** [Fake Reviews Dataset](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset) - Salminen et al. (2020)
- ~40,000 product reviews (OR = genuine, CG = AI-generated)
- Real e-commerce data with ground truth labels

## Architecture

```
For each review:
  1. Embed text → KNN similarity score vs training fakes (Tool 1)
  2. Extract features → XGBoost prediction + confidence (Tool 2)
  3. Decision: If Tool 1 & Tool 2 agree with confidence ≥ 0.65 → Finalize
              Else → Call Judge (Tool 3) for tiebreaker
  4. Log full reasoning trace
```

**Tool Details:**

| Tool | Input | Output | Cost |
|------|-------|--------|------|
| **KNN Similarity** | Review text | Risk score (0-1) vs training fakes | None (local TF-IDF) |
| **Classifier** | TF-IDF embeddings + 9 handcrafted features | Binary prediction, confidence | None (local XGBoost) |
| **Judge** | Review text + similarity score | Verdict + reasoning | Gemini free tier (~$0) |

## Results

Tested on 1,000 reviews (80/20 train/test split, stratified by label):

```
XGBoost Classifier (Tools 1+2, no judge):
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

Judge Routing (cascade logic):
  - Finalized locally:  54.2% (tools agreed, confidence ≥ 0.65)
  - Escalated to judge: 45.8% (disagreement or low confidence)

Note: Judge results not included in headline metrics — Gemini free tier
(20 req/day) was exhausted before the full test set could be evaluated.
Judge API was verified working in a small smoke test.
```

## Installation

```bash
cd MultiAgentAIGCContentRiskPipeline
pip install -r requirements.txt
```

## Usage

### Local Only (no API key needed)
```bash
python main.py
```
Runs Tools 1 & 2. Judge is skipped gracefully; classifier predictions used as final output.

### With Gemini Judge (optional)
```bash
# Add to .env:  GEMINI_API_KEY=your-key
python main.py
```
Enables LLM judge for ambiguous cases. Free tier from https://aistudio.google.com.
Note: free tier is 20 requests/day — sufficient for small batches only.

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
├── eval.py                # Evaluation metrics + baseline
│
├── tools/
│   ├── similarity_tool.py  # TF-IDF + KNN similarity scoring
│   ├── classifier_tool.py  # XGBoost on embeddings + handcrafted features
│   └── judge_tool.py       # Gemini API wrapper (blind judge)
│
├── requirements.txt
└── results/
    ├── metrics.json        # Final metrics
    ├── report.json         # Detailed analysis with baseline comparison
    └── detailed_results.csv
```

## Design Highlights

### 1. Cost-Aware Routing
Only call the expensive judge when:
- Tools 1 & 2 disagree, **OR**
- Tool 2 confidence < 0.65

On the test set, 54.2% of reviews were finalized locally. The judge would only be paid for on the remaining 45.8%.

### 2. Blind LLM Judge
The judge prompt does not include the classifier's prediction — only the review text and the KNN similarity score. This prevents the LLM from anchoring on the classifier output and rubber-stamping it.

### 3. Local-First ML
- TF-IDF embeddings (no OpenAI dependency)
- XGBoost with 9 handcrafted features: length, punctuation density, unique word ratio, uppercase ratio, repetition patterns, and more
- Full pipeline runs offline; Gemini is optional

### 4. Graceful Degradation
- No Gemini key → judge skipped, classifier output used as final prediction
- No Kaggle access → falls back to HuggingFace, then synthetic data
- All failures logged; pipeline always completes

### 5. Observability
- JSON trace logs per decision (tool inputs, outputs, routing reason)
- Per-review predictions + confidence in `detailed_results.csv`
- Baseline comparison included in `report.json`

## Methodology

**Training (80% of data):**
1. Fit TF-IDF vectorizer on training reviews
2. Build KNN index: store training embeddings split by label (fake/genuine)
3. Extract 9 handcrafted features per review
4. Train XGBoost on TF-IDF embeddings + handcrafted features

**Inference (per review):**
1. Tool 1: TF-IDF embed → KNN score (avg distance to k nearest fakes vs genuines)
2. Tool 2: Feature extraction → XGBoost prediction + confidence
3. Routing: agree + confident → finalize; else → judge
4. Log all tool outputs, routing decision, final prediction

**Evaluation:**
- Accuracy, precision, recall, F1, AUROC
- Baseline: TF-IDF (512 features) + Logistic Regression, same train/test split
- Judge call rate tracked separately from final accuracy

## Key Findings

**Finding 1: XGBoost + handcrafted features substantially outperforms baseline**
85.7% accuracy vs. 77.9% for TF-IDF + Logistic Regression on the same data. The gain comes from XGBoost's ability to exploit non-linear interactions between handcrafted features (punctuation density, repetition ratio, unique word ratio) — not from the KNN similarity score, which is used only for routing.

**Finding 2: Routing logic identifies a real uncertain subset**
45.8% of reviews are flagged for judge escalation, meaning the two local tools genuinely disagree or the classifier is uncertain on nearly half the data. This is the subset where an LLM judge adds the most value.

**Finding 3: Handcrafted features match known AI-generation patterns**
Punctuation density, unique word ratio, and repetition are strong signals — consistent with known characteristics of LLM-generated text (templated phrasing, excessive enthusiasm markers).

## Limitations

- **Judge not fully evaluated:** Free tier quota (20 req/day) prevents running the judge on the full test set. A paid tier or batched evaluation across days is needed for complete cascade metrics.
- **Shared embeddings:** Tool 1 and Tool 2 both use the same TF-IDF representation, so they are not fully independent. A future version could use a separate embedding model for Tool 1.
- **Dataset age:** Training data is GPT-2 era; performance on GPT-4/Claude-generated reviews is unknown and likely lower.
- **No cross-validation:** Results are from a single 80/20 split; confidence intervals not computed.

## For Employers / Portfolio

**Highlight:**
> Built a multi-agent fake review detector achieving **85.7% accuracy / 94.5% AUROC** on 1,000 real e-commerce reviews — **+7.8pp over a logistic regression baseline**. Designed a cost-aware cascade that routes only uncertain cases (45.8%) to an LLM judge, with the judge deliberately blinded to prevent anchoring. Demonstrates: multi-agent orchestration, ML pipeline design, cost-aware API routing, graceful degradation, and JSON observability tracing.

**Key Skills Demonstrated:**
- Multi-agent ReAct-style orchestration
- ML pipeline design (feature engineering, XGBoost training, baseline comparison)
- Cost-aware API routing with graceful degradation
- LLM integration (blind judge pattern, retry logic, SDK migration)
- JSON-based observability and tracing

## References

- **Dataset:** [Salminen et al. (2020) - Fake Reviews Dataset](https://osf.io/mryx5/)
- **XGBoost:** https://xgboost.readthedocs.io/
- **Gemini API:** https://ai.google.dev/
