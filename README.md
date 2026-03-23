# Performance Comparison: Apriori vs. FP-Growth 🛒

This repository contains a practical Machine Learning study focused on **Association Rules**. The project compares the efficiency of the classic **Apriori** algorithm against the modern **FP-Growth**, using the *Market Basket Optimization* dataset.

---

## 🚀 Technologies

- **Python 3.12**
- **Pandas** — data manipulation
- **Mlxtend** — association algorithm implementations
- **Kagglehub** — direct Kaggle dataset integration
- **Matplotlib** — result visualization

---

## 🧠 Context & Objective

This project compares two of the most well-known association rule mining algorithms: Apriori and FP-Growth. The goal was to measure execution performance (runtime) and validate result integrity (consistency) using a supermarket transaction dataset.

---

## 🛠️ Challenges & Lessons Learned

Several critical data engineering problems were solved during development:

| Challenge | Solution Applied |
|-----------|-----------------|
| Virtual Environment | Directory error when activating `.venv`. Fixed by correctly creating the environment via `python -m venv` and using relative paths. |
| Dynamic Paths | `kagglehub` downloads files into folders with random hashes. Implemented a directory scan with `os.listdir()` to automatically locate the CSV. |
| Dirty Data | Items arriving as single comma-separated strings instead of lists. Used `.split(",")` and `.strip()` to separate and clean product names. |
| Empty Rules | High support (5%) resulted in empty DataFrames. Fine-tuned `min_support` to 0.5% and 0.1% to capture associations in sparse datasets. |
| Consistency Error | Algorithms returned the same items in different orders. Implemented itemset normalization (converting frozensets to sorted lists and strings) before comparison. |
| Hash Error (Categorical) | Error when sorting lists in Pandas (`unhashable type: 'list'`). Created an `itemset_key` (string) to enable safe sorting and DataFrame comparison. |

---

## 📊 Results

In the tests performed, **FP-Growth** showed clear superiority:

| Algorithm | Runtime |
|-----------|---------|
| Apriori | ~0.0358s |
| FP-Growth | ~0.0041s |

**Verdict:** FP-Growth was approximately **9x faster**, while maintaining full result integrity (Consistency: OK).

---

## 📋 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/apriori-vs-fpgrowth
   ```

2. Create and activate the virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate      # Linux/Mac
   .venv\Scripts\activate         # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the main script:
   ```bash
   python3 main.py
   ```
