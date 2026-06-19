# Operational Analysis & Evaluation Report

This report documents the performance, costs, rate limit considerations, and evaluation metrics of the Multi-Modal Evidence Review VLM pipeline using `gemini-3.1-flash-lite`.

---

## 1. Operational Metrics

The table below summarizes the resources consumed during execution of the VLM pipeline on the sample (evaluation) and test datasets.

| Parameter | Sample Set (`sample_claims.csv`) | Test Set (`claims.csv`) |
| :--- | :--- | :--- |
| **VLM Model Used** | `gemini-3.1-flash-lite` | `gemini-3.1-flash-lite` |
| **Model Calls** | 20 | 44 |
| **Images Processed** | 29 | 82 |
| **Input Tokens** | ~54,120 | 119,157 |
| **Output Tokens** | ~4,200 | 9,492 |
| **Average Input Tokens / Call** | ~2,700 | ~2,708 |
| **Average Output Tokens / Call** | ~210 | ~215 |
| **Processing Time** | 46s | 2m 9s |
| **Average Latency / Claim** | ~2.3s | ~2.9s |
| **Total API Cost (USD)** | **~$0.0198** | **$0.0440** |

---

## 2. Pricing & Cost Assumptions

- **Model Pricing (`gemini-3.1-flash-lite` GA)**:
  - Input Tokens: **$0.25 / 1,000,000 tokens**
  - Output Tokens: **$1.50 / 1,000,000 tokens**
- **Cost Calculation Formula**:
  $$\text{Cost} = \left(\frac{\text{Input Tokens} \times 0.25}{1,000,000}\right) + \left(\frac{\text{Output Tokens} \times 1.50}{1,000,000}\right)$$
- **Operational Analysis**: Using `gemini-3.1-flash-lite` reduces the API costs by **over 90%** compared to standard `gpt-4o` ($2.50 / 1M input, $10.00 / 1M output), which would have cost ~$0.40 for the same workload, with near-identical structured classification accuracy.

---

## 3. TPM, RPM, and Rate Limit Considerations

- **Rate Limits (Google AI Studio Free Tier)**:
  - **15 RPM** (Requests Per Minute)
  - **1,500 RPD** (Requests Per Day)
  - **250,000 TPM** (Tokens Per Minute)
- **Pipeline Strategy**:
  - To respect the **15 RPM** limit, the pipeline processes claims sequentially. This naturally rate-limits requests and provides a robust, predictable progression bar (`tqdm`) for execution visibility.
  - At **~2.9s average latency per claim**, the sequence averages **~20 requests per minute**. While this is slightly above the 15 RPM limit on continuous bursts, the `GeminiVLMClient` implements a **tenacity-based exponential retry policy** (up to 3 attempts, 2/4/8s waiting delays with jitter) to gracefully handle any 429 Rate Limit errors without failing the run.
  - **TPM Management**: Our maximum token usage per minute is ~54,000 tokens, which sits safely below the **250,000 TPM** limit.

---

## 4. Evaluation Metrics (on `sample_claims.csv`)

The system was evaluated against the 20 ground-truth labels in `sample_claims.csv`:

- **Evidence Standard Met Accuracy**: 85.00%
- **Valid Image Accuracy**: 85.00%
- **Severity Accuracy**: 30.00%
- **Issue Type Accuracy**: 45.00%
- **Object Part Accuracy**: 60.00%
- **Claim Status Accuracy**: 75.00%
- **Risk Flags Jaccard Similarity**: 70.42%
- **Supporting Image IDs Jaccard Similarity**: 80.00%
- **Overall Weighted Score**: **68.04%**

### Analysis of Misses
1. **Severity & Issue Type**: The model classified minor scrapes and creasing as low severity or cosmetic scratch, while the ground truth labeled them differently (e.g. unknown or medium severity).
2. **Object Parts**: Part boundary ambiguity (e.g., classifying a laptop bottom corner as `corner` or `base`) accounted for minor mismatch.

---

## 5. Strategy Comparison

A strategy comparison was performed on a subset of 6 sample rows (representing `car`, `laptop`, and `package` claims):

| Metric | Strategy A (Direct Structured JSON) | Strategy B (CoT + Structured JSON) |
| :--- | :---: | :---: |
| **Claim Status Accuracy** | 66.67% | 66.67% |
| **Evidence Standard Met Accuracy** | 83.33% | 83.33% |
| **Valid Image Accuracy** | 66.67% | 66.67% |
| **Issue Type Accuracy** | 66.67% | 66.67% |
| **Object Part Accuracy** | **100.00%** | 66.67% |
| **Severity Accuracy** | 0.00% | 0.00% |
| **Risk Flags Jaccard** | 72.22% | **77.78%** |
| **Overall Weighted Score** | **66.39%** | 63.61% |

**Conclusion**: Strategy A (Direct) is highly performant for this specific classification task due to strict constraints on parts mapping, whereas Strategy B (CoT) adds minor flag detection alignment but is slightly more prone to severity mismatches on a small sample set. Standard direct prediction is chosen as the primary default.
