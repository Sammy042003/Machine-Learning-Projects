# Zomato Restaurant Reviews – Sentiment Analysis (NLP & Machine Learning)

## 📌 Project Overview
This project focuses on performing **sentiment analysis on Zomato restaurant reviews** using **Natural Language Processing (NLP)** and **Machine Learning** techniques. The goal is to automatically classify customer reviews as **positive or negative**, enabling businesses to understand customer sentiment at scale and make data-driven decisions.

The project follows a complete **end-to-end data science pipeline**, covering data understanding, preprocessing, exploratory data analysis, hypothesis testing, feature engineering, model building, evaluation, and explainability.

---

## 🎯 Business Objective
Online reviews significantly influence customer decisions and restaurant reputation. Manually analyzing large volumes of reviews is inefficient and not scalable.

This project aims to:
- Automate sentiment classification of customer reviews
- Identify patterns in customer satisfaction and dissatisfaction
- Help restaurants improve service quality using actionable insights
- Enable scalable customer feedback analytics

---

## 📂 Dataset Description
The project uses two datasets:

1. **Zomato Restaurant Names and Metadata**
   - Contains restaurant-related metadata

2. **Zomato Restaurant Reviews**
   - Restaurant name
   - Reviewer details
   - Review text
   - Rating
   - Timestamp and metadata

The **review dataset** is primarily used for sentiment analysis.

---

## 🛠️ Tech Stack & Libraries
- **Programming Language:** Python  
- **Libraries Used:**
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - nltk
  - scikit-learn
  - imbalanced-learn

---

## 🔍 Project Workflow

### 1. Data Understanding
- Loaded datasets and inspected structure
- Checked data types, missing values, and duplicates
- Analyzed basic statistics and data distribution

---

### 2. Data Wrangling
- Removed duplicate records
- Handled missing values
- Ensured correct data types
- Cleaned inconsistent entries

---

### 3. Exploratory Data Analysis (EDA)
- Performed **15 meaningful visualizations**
- Followed **UBM Rule**:
  - Univariate Analysis
  - Bivariate Analysis
  - Multivariate Analysis
- Extracted business-relevant insights from each chart

---

### 4. Hypothesis Testing
- Defined **three hypotheses** based on EDA findings
- Applied appropriate statistical tests
- Evaluated p-values
- Drew statistically valid conclusions

---

### 5. Textual Data Preprocessing (NLP)
Applied comprehensive NLP preprocessing:
- Contraction expansion
- Lowercasing
- Removing punctuation, URLs, digits
- Stopword removal
- Tokenization
- Lemmatization
- POS tagging

---

### 6. Feature Engineering & Vectorization
- Created binary sentiment labels from ratings
- Converted text into numerical features using **TF-IDF Vectorization**
- Reduced noise and improved model performance

---

### 7. Handling Imbalanced Dataset
- Checked class distribution
- Applied balancing techniques where required
- Ensured unbiased model learning

---

### 8. Machine Learning Models
Implemented and compared three models:
1. **Logistic Regression**
2. **Multinomial Naive Bayes**
3. **Additional Classification Model**

For each model:
- Training and prediction
- Evaluation using Accuracy, Precision, Recall, F1-score
- Cross-validation and hyperparameter tuning
- Performance comparison before and after tuning

---

### 9. Model Evaluation & Selection
- Logistic Regression selected as final model
- Best balance of precision, recall, and F1-score
- Strong generalization and interpretability

---

### 10. Model Explainability
- Analyzed feature importance
- Identified influential words affecting sentiment
- Improved transparency and business trust

---

## 📊 Evaluation Metrics Used
- Accuracy
- Precision
- Recall
- F1-score

These metrics ensure the model minimizes incorrect sentiment classification, especially important for customer satisfaction analysis.

---

## ✅ Final Outcome
- Built a scalable sentiment analysis system
- Automated customer feedback interpretation
- Provided actionable insights for business improvement
- Delivered a production-ready, interpretable ML solution

---

## 🚀 Future Improvements
- Deploy model using a web application
- Extend to multi-class sentiment analysis
- Use deep learning models (LSTM, BERT)
- Real-time sentiment monitoring dashboard

---

## 👤 Author
**Smyan**

---

## 📌 Note
This project notebook is fully executable end-to-end without errors and follows structured documentation and modular coding standards.
