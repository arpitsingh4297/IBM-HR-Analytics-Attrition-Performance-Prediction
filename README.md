# IBM-HR-Analytics-Attrition-Performance-Prediction

Based on your project code and structure, here's a professional and comprehensive **README** file tailored for GitHub for your project **“Employee Performance & Attrition Prediction (HR Analytics)”**:

---

# 🧠 Employee Performance & Attrition Prediction (HR Analytics)

A comprehensive machine learning project using IBM HR Analytics dataset to predict employee attrition and understand performance drivers. This solution leverages data preprocessing, model training, SHAP explainability, and actionable business insights to enable data-driven HR decisions.

---

## 📁 Project Structure

```
📦 Employee-Attrition-Performance-Prediction/
│
├── 📊 EDA & Preprocessing
│   └── Data cleaning, null check, encoding, outlier handling
│
├── 🤖 ML Modeling
│   └── Model training, SMOTE balancing, evaluation metrics
│
├── 🔍 Model Tuning
│   └── Hyperparameter tuning using GridSearchCV
│
├── 🌐 SHAP Explainability
│   └── Global and local SHAP plots for business interpretation
│
├── 📈 Streamlit Dashboard (optional)
│   └── Interactive visual dashboard for deployment (TBD)
│
├── 📜 business_report.pdf/.docx
│   └── Business insights, executive summary, recommendations
│
└── 📂 data/
    └── WA_Fn-UseC_-HR-Employee-Attrition.csv
```

---

## 📌 Project Objectives

- Predict employee **attrition** using ML models.
- Identify top **factors** influencing attrition and performance.
- Improve HR policies with **data-backed** evidence.
- Enable **explainable AI** for decision transparency.

---

## 🛠️ Tools & Libraries Used

- **Python, Pandas, NumPy, Seaborn, Matplotlib**
- **Scikit-learn** – modeling, preprocessing
- **XGBoost, Random Forest, Naive Bayes, AdaBoost**
- **SMOTE** – handle class imbalance
- **SHAP** – model interpretability
- **Streamlit** – (for dashboard visualization – optional)

---

## 🔎 Exploratory Data Analysis (EDA)

- Checked data types, distributions, and outliers.
- Visualized categorical and numerical features.
- Explored key relationships between attrition and HR features like:
  - Job role
  - Monthly income
  - OverTime
  - Work-life balance
  - Job satisfaction

---

## 📊 Machine Learning Models

Evaluated the following models on both **original** and **SMOTE-balanced** data:

- Decision Tree Classifier
- Random Forest Classifier
- Naive Bayes
- AdaBoost
- XGBoost

**Evaluation Metrics:**

- Precision, Recall, F1 Score, ROC AUC (Train/Test)

Best models selected based on **Test F1 Score**.

---

## 🧪 Model Tuning

- Used **GridSearchCV** for GaussianNB to optimize `var_smoothing`.
- Identified best hyperparameters and improved test metrics.

---

## 💡 SHAP-Based Explainability

- Visualized **global** SHAP summary to identify top drivers of attrition.
- Generated **individual waterfall plots** for specific employees.
- Key insights:
  - Employees with frequent overtime and low job satisfaction are more likely to leave.
  - Years with current manager and work-life balance have strong influence on retention.

---

## 💼 Business Recommendations

1. **Improve Work-Life Balance:** Reduce forced overtime and promote flexible work.
2. **Boost Job Satisfaction:** Focus on training, career path clarity, and recognition.
3. **Track At-Risk Employees:** Use model predictions and SHAP to trigger HR interventions.
4. **Retention Policies:** Design retention programs targeting high attrition clusters.

---

## 📈 Dashboard (Optional)

This project is extendable to an interactive **Streamlit dashboard** for real-time HR analytics. (Coming soon...)

---

## 📁 Dataset Source

IBM HR Analytics Employee Attrition & Performance Dataset  
🧾 [Kaggle Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

---

## 👤 Author

**Arpit Singh**  
Data Scientist & Business Analyst Aspirant  
📧 arpitdatasci@gmail.com | 🐙 [GitHub: @arpitsingh4297](https://github.com/arpitsingh4297)

---

## 📌 License

This project is licensed under the MIT License.  
Feel free to fork, use, and customize.

---

Would you like a **downloadable README.md file** or shall I directly add formatting like badges, emojis, and links for a more visually aesthetic GitHub display?
