# AI Ethics and Fairness in Machine Learning

## 📌 Overview
This project focuses on **AI Ethics and Bias Mitigation** in machine learning models. Using **Fairlearn**, it evaluates and reduces bias in predictions based on demographic factors. The model is trained on the **UCI Adult dataset** to predict income while ensuring fairness.

## 📂 Project Structure
- **`aiethics.py`** – The main script for data preprocessing, model training, bias detection, and mitigation.

## 🔧 Key Features
✔️ **Preprocessing**: Cleans and encodes categorical & numeric features for model training.  
✔️ **Logistic Regression Model**: Predicts income categories based on demographic attributes.  
✔️ **Bias Detection**: Evaluates fairness using demographic parity metrics.  
✔️ **Bias Mitigation**: Implements **Exponentiated Gradient** to reduce model bias.  
✔️ **Evaluation Metrics**: Measures **accuracy, fairness, and trade-offs** for transparent AI decision-making.  

## 📊 Dataset Information
The dataset used is the **UCI Adult Income Dataset** from OpenML:
- **Features**: Age, education, occupation, race, gender, and financial attributes.
- **Target Variable**: Income classification (`>50K` or `<=50K` per year).
- **Sensitive Attribute**: `sex` (used to evaluate fairness in predictions).

## 📈 Model Training & Fairness Evaluation
- **Preprocessing**: Transforms categorical & numeric features for machine learning.
- **Model Training**: Uses **Logistic Regression** to predict income.
- **Fairness Metrics**: Calculates **demographic parity difference & ratio** to measure bias.

### **Example Output (Before Bias Mitigation)**
```
Initial Model Accuracy: 0.85
Demographic Parity Difference: 0.18
Demographic Parity Ratio: 0.72
```

## ⚖️ Bias Mitigation using Exponentiated Gradient
After applying fairness constraints, bias is reduced:
```
Mitigated Model Accuracy: 0.83
Mitigated Demographic Parity Difference: 0.09
Mitigated Demographic Parity Ratio: 0.91
```

## 🚀 Future Enhancements
- Implement alternative fairness constraints such as **Equalized Odds**.
- Deploy as a **web API** for real-time fairness auditing.
- Investigate multiple sensitive attributes beyond `sex`.
- Optimize trade-offs between fairness and model accuracy.

## 📝 License
This project is licensed under the **MIT License**.

---

### 🔗 **Connect with Me**
💼 GitHub: [Nimra71](https://github.com/Nimra71)  
📧 Email: [nimrafatima745@gmail.com]  

---

