# Multi-Modal Evidence Review Pipeline

This directory contains the Python codebase to analyze insurance damage claims using `gemini-3.1-flash-lite` VLM API. It processes text conversation transcripts, user risk history, evidence requirements, and images to determine claim supportability, severity, parts, issue types, and fraud risk flags.

---

## 📂 Codebase Layout

```text
code/
├── main.py                    # Terminal CLI entry point
├── pipeline.py                # Pipeline orchestrator
├── vlm_client.py              # Gemini VLM Client wrapper (Tenacity retry + Pydantic schema)
├── prompt_builder.py          # Structured system/user prompts construction
├── safety.py                  # Prompt-injection detection and sanitization
├── postprocessor.py           # Enum compliance, overrides, and flag merges
├── requirements.txt           # Python library dependencies
├── .env.example               # Template for environment variables
└── evaluation/
    ├── main.py                # Evaluation suite entry point (sample set accuracy)
    ├── metrics.py             # Math helpers (Accuracies & Jaccard flags similarity)
    ├── compare_strategies.py  # A/B testing of Direct vs CoT methods
    └── evaluation_report.md   # Operational analysis report (costs, TPM, rate limits)
```

---

## 🛠️ Execution & Setup

Refer to the root-level [run_instruction.md](file:///d:/hackerearth/run_instruction.md) or [run instruction.md](file:///d:/hackerearth/run%20instruction.md) for full commands.

1. **Virtual Environment**:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\activate
   ```

2. **Install requirements**:
   ```powershell
   pip install -r code/requirements.txt
   ```

3. **Set API Key**:
   Create a `code/.env` file with your Gemini API key:
   ```env
   GOOGLE_API_KEY=your_actual_gemini_api_key
   ```

4. **Run Predictions**:
   ```powershell
   python code/main.py
   ```
   *This reads `dataset/claims.csv` and outputs predictions to `output.csv` in the root directory.*

5. **Run Evaluation**:
   ```powershell
   python code/evaluation/main.py
   ```

6. **Compare Strategies**:
   ```powershell
   python code/evaluation/compare_strategies.py
   ```

---

## 🧠 Architecture Detail

### 1. Pre-Processing Safety Layer (`safety.py`)
Scans the user claim conversation text for prompt-injection keyword sequences (e.g., "approve immediately", "ignore previous instructions", "skip manual review"). It flags attempts as `text_instruction_present` in risk flags and sanitizes the claim text.

### 2. Prompt Builder (`prompt_builder.py`)
Separates instructions from data using clear brackets and context descriptors. It dynamically maps applicable checklists from `evidence_requirements.csv` into the user prompt based on keyword scans.

### 3. VLM Client (`vlm_client.py`)
Uses the `google-genai` SDK and passes a `pydantic` structure matching the task constraints to enforce token-level JSON formatting, ensuring zero output formatting failures. It wraps execution with `tenacity` exponential backoff retries to handle 429 rate-limiting.

### 4. Post-Processor (`postprocessor.py`)
Forces enums to valid choices (dent, scratch, screen, hinge, front_bumper, package_corner, etc.), ensures boolean columns match lowercase `"true"`/`"false"` specifications, and applies conditional logic rules (e.g., if images are invalid, the evidence standard cannot be met).
