# Translation Model Comparison (English ↔ Hindi)

## 📌 Project Overview

This project compares multiple machine translation models by translating English sentences to Hindi and then back to English.

The goal is to evaluate how well different models preserve meaning using similarity metrics.

---

## 🤖 Models Used

* TranslateGemma
* MarianMT
* NLLB (No Language Left Behind)
* mBART (optional extension)

---

## 🧪 Methodology

1. Translate English → Hindi
2. Translate Hindi → English (back-translation)
3. Compare original and back-translated sentences
4. Evaluate using:

   * Cosine Similarity
   * Accuracy (threshold-based)

---

## 📊 Evaluation Metrics

### 🔹 Cosine Similarity

Measures semantic similarity between original and translated text.

### 🔹 Accuracy

Defined as:

* 1 if similarity ≥ 0.80
* 0 otherwise

---

## 📈 Results

| Model          | Avg Similarity | Accuracy (%) |
| -------------- | -------------- | ------------ |
| TranslateGemma | 0.25           | 12.5%        |
| MarianMT       | 0.81           | 62.0%        |
| NLLB           | 0.87           | 76.5%        |

👉 NLLB performed the best overall.

---

## 📂 Project Structure

* `comparison.ipynb` → main evaluation notebook
* `*_results.xlsx` → model outputs
* `*_bt.xlsx` → back-translation results

---

## ⚠️ Note

Large datasets are not included due to GitHub size limitations.

---

## 🚀 Future Improvements

* Add BLEU / ROUGE metrics
* Train custom translation model
* Use larger datasets
* Optimize translation parameters

---

## 👩‍💻 Author

Bhavana
MS Data Science — Clarkson University
