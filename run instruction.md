# Run Instructions

This document provides step-by-step instructions to set up the environment, run the multi-modal evidence review pipeline, and execute the evaluation suite.

## 1. Prerequisites

- Python 3.10+ (Windows standard `py` launcher)
- Active internet connection (for Gemini VLM API calls)

## 2. Environment Setup

1. **Create the virtual environment**:
   ```powershell
   py -m venv .venv
   ```

2. **Install required packages**:
   ```powershell
   .\.venv\Scripts\pip install -r code/requirements.txt
   ```

3. **Configure API Keys**:
   Create a `.env` file inside the `code/` folder:
   ```env
   # code/.env
   GOOGLE_API_KEY=your_actual_gemini_api_key
   ```
   *Note: Both `GOOGLE_API_KEY` and `GEMINI_API_KEY` are supported. The API key is loaded dynamically at execution time.*

## 3. Running predictions on the test dataset

To execute the VLM processing pipeline on the test claims file (`dataset/claims.csv`) and output predictions to the root `output.csv` file:

```powershell
.\.venv\Scripts\python.exe code/main.py
```

*By default, the script reads `dataset/claims.csv` and outputs to `output.csv` in the root directory.*

### CLI Options:
- `--input`: Override path to the input CSV file. (default: `dataset/claims.csv`)
- `--output`: Override path for the output CSV file. (default: `output.csv`)
- `--dataset-root`: Path to the dataset folder. (default: `dataset`)

Example:
```powershell
.\.venv\Scripts\python.exe code/main.py --input dataset/claims.csv --output output.csv
```

## 4. Running the evaluation suite

### A. Run Main Evaluation
Evaluate the VLM pipeline against the labeled ground-truth sample dataset (`dataset/sample_claims.csv`) to compute accuracy, severity correctness, and Jaccard similarity for risk flags:

```powershell
.\.venv\Scripts\python.exe code/evaluation/main.py
```
*Results will print to the console and be saved to `code/evaluation/last_evaluation_results.txt`.*

### B. Compare Strategies
Compare **Strategy A (Direct Structured JSON)** versus **Strategy B (Chain-of-Thought reasoning + Structured JSON)** on a representative subset of sample claims:

```powershell
.\.venv\Scripts\python.exe code/evaluation/compare_strategies.py
```
*Outputs a side-by-side metric comparison table directly to the console.*
