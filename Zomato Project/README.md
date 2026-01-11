🍽️ Zomato Restaurant Reviews – Sentiment Analysis (ML & NLP Project)
📌 Project Overview

This project focuses on performing sentiment analysis on Zomato restaurant reviews using Natural Language Processing (NLP) and Machine Learning techniques. The goal is to automatically classify customer reviews into positive or negative sentiment, enabling businesses to understand customer opinions and improve decision-making.

The project follows a complete end-to-end data science pipeline, including data understanding, preprocessing, exploratory data analysis (EDA), feature engineering, visualization, hypothesis testing, and multiple machine learning model implementations.

🎯 Business Objective

Analyze customer feedback at scale

Identify patterns in customer sentiment

Support restaurants in improving service quality

Enable data-driven business decisions using customer reviews

📂 Dataset Description

The project uses two datasets:

Restaurant Metadata Dataset

Restaurant name

Location and other attributes

Restaurant Reviews Dataset

Reviewer name

Review text

Ratings

Review time and metadata

🔧 Technologies & Libraries Used

Programming Language: Python

Data Handling: Pandas, NumPy

Visualization: Matplotlib, Seaborn

NLP: NLTK, Regex

Feature Extraction: TF-IDF Vectorizer

Machine Learning: Scikit-learn

Model Evaluation: Accuracy, Precision, Recall, F1-score

🧠 Project Workflow
1️⃣ Know Your Data

Dataset loading

Dataset overview (rows, columns, data types)

Handling missing values and duplicates

2️⃣ Understanding Variables

Column analysis

Statistical summary

Unique value inspection

3️⃣ Data Wrangling

Data cleaning

Handling inconsistent and missing values

Preparing dataset for analysis

4️⃣ Exploratory Data Analysis (EDA)

15 meaningful visualizations following UBM Rule

Univariate analysis

Bivariate analysis

Multivariate analysis

Insights extracted from each chart

Business impact discussion for each visualization

5️⃣ Hypothesis Testing

Three hypotheses were defined and tested using appropriate statistical tests:

Null and alternate hypothesis formulation

P-value computation

Statistical conclusions with business interpretation

🛠️ Feature Engineering & Data Preprocessing
Textual Data Preprocessing (NLP)

Expanding contractions

Lowercasing

Removing punctuation

Removing URLs and digits

Stopword removal

Tokenization

Lemmatization

POS tagging

Text vectorization using TF-IDF

Feature Manipulation & Selection

Feature reduction to avoid overfitting

Selection of important features

Data Transformation & Scaling

Applied where required for model performance

Handling Imbalanced Dataset

Checked class distribution

Applied balancing techniques when necessary

🤖 Machine Learning Models Implemented
🔹 Model 1: Logistic Regression

Strong performance on text classification

Balanced precision and recall

Selected as the final prediction model

🔹 Model 2: Multinomial Naive Bayes

Efficient for text-based data

Fast training and inference

Stable performance with minimal tuning

🔹 Model 3: Additional Classification Model

Used for comparison and robustness analysis

Each model includes:

Model training

Prediction

Evaluation metric visualization

Cross-validation and hyperparameter tuning

Business impact analysis

📊 Evaluation Metrics Used

Accuracy

Precision

Recall

F1-Score

These metrics were selected to ensure:

Balanced sentiment detection

Reduced false positives and false negatives

Business-aligned decision-making

🏆 Final Model Selection

Logistic Regression was chosen as the final model due to:

Best overall F1-score

High interpretability

Stable performance on unseen data

Strong alignment with business objectives

📈 Business Impact

Automated sentiment classification

Faster analysis of customer feedback

Identification of service improvement areas

Scalable solution for large datasets

📌 Conclusion

This project delivers a production-ready sentiment analysis pipeline using real-world data. It combines robust NLP preprocessing, insightful visual analysis, and reliable machine learning models to create a scalable and interpretable solution for customer sentiment analytics.
