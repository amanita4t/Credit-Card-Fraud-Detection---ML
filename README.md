# Credit Card Fraud Detection

## Project Overview

The goal of this project is to develop a **machine learning model** that can accurately detect fraudulent credit card transactions using historical data. By analyzing transaction patterns, the model aims to distinguish between **normal** and **fraudulent** activity, helping financial institutions flag suspicious behavior early and reduce potential risks.

---

## Challenges

- **Imbalanced dataset:** Fraud cases are a very small fraction of total transactions.
- **High precision required:** Minimize false positives (flagging a valid transaction as fraud).
- **High recall required:** Maximize detection of fraudulent cases.

---

## Dataset

- **Source:** `creditcard.csv`
- **Features:**

  - `V1–V28`: PCA-transformed anonymized features to protect sensitive information.
  - `Amount`: Transaction amount.
  - `Class`: Target variable (`0` = normal, `1` = fraud).

## Sample File Access

You can download a sample of the dataset here: https://drive.google.com/file/d/1SStXExMWoTgxhXXl_scN6O1WOuGjFE5W/view?usp=drive_link

**Fraud distribution:**

- Fraud cases are less than **0.2%** of total transactions, highlighting the imbalance.

---

## Approach

1. **Data Exploration**

   - Checked fraud vs normal transactions.
   - Analyzed feature distributions.
   - Visualized feature correlations using a heatmap.

2. **Preprocessing**

   - Separated input features (`X`) and target (`y`).
   - Performed train-test split (80/20).

3. **Model Training**

   - Used a **RandomForestClassifier**.
   - Trained on the training dataset.

4. **Model Evaluation**

   - Predictions made on the test set.
   - Metrics computed: Accuracy, Precision, Recall, F1-score, MCC.
   - Visualized results with a confusion matrix.

---

## Model Evaluation Metrics

The model accuracy is high due to class imbalance, so we focus on **precision, recall, and F1 score** for better insights.

- **Accuracy: 0.9996** → 99.96% correct predictions. High but misleading in imbalanced datasets.
- **Precision: 0.9740** → When predicting fraud, it was correct 97.4% of the time. Very few false alarms.
- **Recall: 0.7653** → Detected 76.53% of all fraud cases. Some frauds were missed.
- **F1-Score: 0.8571** → Balanced score between precision and recall. Strong performance overall.
- **MCC: 0.8632** → Balanced metric for imbalanced datasets. Indicates strong predictive power.

---

## Confusion Matrix

The confusion matrix provides a breakdown of true positives, true negatives, false positives, and false negatives.

- **True Negatives (TN):** Correctly classified normal transactions.
- **False Positives (FP):** Normal transactions incorrectly flagged as fraud.
- **False Negatives (FN):** Fraud cases missed by the model.
- **True Positives (TP):** Fraud cases correctly detected.

---

## How to Run the Project

1. Clone this repository.
2. Install dependencies:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

3. Place `creditcard.csv` in the project directory.
4. Run the Jupyter Notebook:

   ```bash
   jupyter notebook main.ipynb
   ```

---

## Key Takeaways

- The Random Forest model achieved **high precision** (very few false alarms).
- **Recall** can be improved to catch more fraud cases.
- Metrics like **MCC** and **F1-score** provide a more realistic view than accuracy alone.

This project demonstrates how machine learning can effectively address **fraud detection challenges** in highly imbalanced datasets.
