# Individual Research Project

# 🔬 Explainable AI Fraud Detection System

Final Year Individual Research Project: A comprehensive machine learning system for fraud detection with explainable AI capabilities, featuring multiple models, streaming analysis, cross-validation analysis, and an interactive web application.


# 🎯 Overview

This project presents a state-of-the-art fraud detection system that combines multiple machine learning approaches with explainable AI (XAI) techniques. The system is designed to detect fraudulent transactions in real-time while providing clear explanations for its decisions, making it suitable for production deployment in financial institutions.


# 🔬 Research Objectives

    - Compare model architectures for fraud detection (XGBoost vs. Ensemble methods).

    - Implement explainable AI using SHAP (Shapley Additive Explanations) for model transparency.

    - Develop a production-ready system with real-time capabilities via a Streamlit web application.

    - Evaluate performance robustly using stratified cross-validation on real-world imbalanced datasets.

    - Provide actionable insights by bridging the gap between academic model performance and real-world business impact.


# ✨ Key Features

    🤖 Machine Learning Models: XGBoost Classifiers, weighted Voting Ensembles (Random Forest + Logistic Regression), and advanced Stacking Classifiers.

    ⚙️ In-App Cross-Validation: Perform comprehensive, stratified k-fold cross-validation directly within the dashboard to get unbiased performance estimates.

    🔍 Explainable AI: Instance-level explanations via SHAP waterfall plots, global feature importance analysis, and business-friendly interpretation guides.

    📊 Interactive Web Application: A multi-page Streamlit app for real-time simulation, batch file analysis, and deep-dive model comparisons.

    📈 Advanced Analytics: ROC/PR curves, confusion matrices, prediction score distributions, threshold analysis, and statistical significance testing (Wilcoxon signed-rank test).

    💼 Business Impact Assessment: Translates model metrics (Recall, Precision) into estimated financial impact, including fraud value saved and estimated investigation costs.



# 📊 Datasets

    Credit Card Fraud Dataset

        Source: Kaggle

        Size: 284,807 transactions

        Features: 30 anonymised PCA features (V1-V28), Time, Amount

        Fraud Rate: 0.172% (highly imbalanced)

        Use Case: Real-world credit card transaction analysis.

    PaySim Synthetic Dataset

        Source: Kaggle

        Size: 6.3M transactions (a 280k sample is used for app performance)

        Features: Transaction type, amount, account balances, errorBalance features.

        Fraud Rate: ~0.13% (realistic simulation)

        Use Case: Mobile payment and money transfer fraud simulation.


# 🧠 Models Implemented

The system evaluates several model architectures, defined within the Streamlit application.

    XGBoost Classifiers:
    '''markdown
    '''python
    XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=ratio,  # Handles class imbalance
        use_label_encoder=False,
        eval_metric='logloss'
    ) 
    
    Enasemble Methods (Voting):
    '''markdown
    '''python
    VotingClassifier(
    estimators=[('rf', RandomForestClassifier), ('lr', LogisticRegression)],
    voting='soft',
    weights=[0.7, 0.3]  # Optimised weights
    )


    Stacking Classifiers
    '''markdown
    '''python
    StackingClassifier(
        estimators=[('rf', RandomForestClassifier), ('xgb', XGBoostClassifier)],
        final_estimator=LogisticRegression(),
        passthrough=True  # Includes original features for the meta-model
    )


# 🚀 Quick Start Guide
Prerequisites

Python 3.8+ installed on your computer
Git installed (to clone the repository)
Git LFS (to handle large files)
Internet connection (to download dependencies)

1. Clone the Repository

git clone https://github.com/brandonelward/Individual-Research-Project.git
cd Individual-Research-Project

2. Set Up Python Environment
On Windows:

python -m venv .venv
.venv\Scripts\activate

On Mac/Linux:

python3 -m venv .venv
source .venv/bin/activate

3. Install Required Packages

pip install -r requirements.txt

4. Download Required Data Files
You need to add the datasets to the data/ folder. Create the folder if it doesn't exist:
Required files:

data/creditcard.csv - Download from Kaggle - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud 

data/transactions.csv - Download from Kaggle - https://www.kaggle.com/datasets/ealaxi/paysim1 - Rename csv to transactions.csv

mkdir data
# Place your downloaded CSV files in the data/ folder

5. Train Individual Models

# Run Jupyter notebooks in sequence:
jupyter notebook notebooks/01_creditcard_xgboost_model.ipynb
jupyter notebook notebooks/02_transactions_xgboost_model.ipynb
jupyter notebook notebooks/03_creditcard_ensemble__model.ipynb
jupyter notebook notebooks/04_transactions_ensemble_model.ipynb

6. Run Cross-Validation Analysis

python run_cv_analysis.py

7. Run the Dashboard

streamlit run app/app.py
or
python -m streamlit run app/app.py

8. Open Your Browser

The dashboard will automatically open at: http://localhost:8501
If it doesn't open automatically, copy and paste this URL into your browser.

# 🔧 Troubleshooting
Problem: ModuleNotFoundError
Solution: Make sure your virtual environment is activated and run pip install -r requirements.txt again

Problem: FileNotFoundError for datasets
Solution: Ensure CSV files are in the data/ folder with exact names: creditcard.csv and transactions.csv

Problem: Port already in use
Solution: Run streamlit run app/app.py --server.port 8502 to use a different port

Problem: Python command not found
Solution: Try python3 instead of python, or install Python from python.org

# 💻 Usage
Running the Web Application 

All analysis is performed through the interactive web interface.


# 🖥️ Web Application Features

    Single Model Analysis

        Batch Analysis: Upload a CSV file for bulk fraud detection and receive an annotated output.

        Real-Time Simulation: Stream sample transactions with configurable speed to simulate a live feed.

        SHAP Explanations: View detailed waterfall plots to understand exactly why a transaction was flagged.

    Model Comparison

        Side-by-side metrics: Compare Ensemble and XGBoost models on key metrics like AUPRC and F1-Score.

        ROC and PR curves: Interactively analyse the precision-recall trade-off.

        Feature importance: See which features are most influential for each model.

    Cross-Validation Analysis

        On-the-fly execution: Run a full 5-fold stratified CV on multiple models directly in the app.

        Statistical significance: View Wilcoxon signed-rank test results to confirm if performance differences are statistically significant.

        Automated recommendation: The app automatically recommends the best model based on a weighted score of key performance metrics.


# 📈 Model Performance

Performance metrics are based on the hold-out test set from the training notebooks.

Single Model performance Summary:

Creditcard Dataset Results:
This table shows the performance metrics when the models are evaluated on the full creditcard.csv dataset.

| Model | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) |
| :--- | :---: | :---: | :---: | :---: |
| **Ensemble (RF+LR)** | **0.975** | 0.939 | 0.957 | 0.971 |
| **XGBoost** | 0.973 | **0.967** | **0.970** | **0.978** |


Transactions Dataset Results:

This table shows the performance metrics when the models are evaluated on the transactions_sample_280k.csv dataset.

| Model | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) |
| :--- | :---: | :---: | :---: | :---: |
| **Ensemble (RF+LR)** | **1.000** | 0.997 | 0.998 | **0.999** |
| **XGBoost** | 0.998 | **0.999** | **0.999** | **0.999** |

Cross-Validation Performance Summary:


Credit Card Dataset:

These tables shows the average performance and standard deviation across 5 folds of stratified cross-validation. This is a robust measure of how the models are expected to perform on unseen data.

| Model | Accuracy | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.9765 ± 0.0015 | 0.0632 ± 0.0032 | **0.9106 ± 0.0236** | 0.1182 ± 0.0055 |
| Random Forest | **0.9995 ± 0.0001** | **0.9593 ± 0.0287** | 0.7581 ± 0.0207 | 0.8468 ± 0.0212 |
| XGBoost | **0.9996 ± 0.0001** | 0.9158 ± 0.0370 | 0.8293 ± 0.0236 | **0.8700 ± 0.0252** |
| Stacking Ensemble (RF+XGB) | **0.9995 ± 0.0000** | 0.9590 ± 0.0256 | 0.7419 ± 0.0185 | 0.8362 ± 0.0109 |


Transactions Dataset:

| Model | Accuracy | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.6829 ± 0.0066 | 0.0826 ± 0.0016 | 0.9779 ± 0.0024 | 0.1523 ± 0.0028 |
| **Random Forest** | **0.9999 ± 0.0000** | **1.0000 ± 0.0000** | 0.9957 ± 0.0016 | **0.9978 ± 0.0008** |
| XGBoost | 0.9998 ± 0.0000 | 0.9953 ± 0.0020 | **0.9962 ± 0.0015** | 0.9958 ± 0.0007 |
| Stacking Ensemble (RF+XGB) | 0.9881 ± 0.0003 | 0.9306 ± 0.0069 | 0.6403 ± 0.0122 | 0.7585 ± 0.0073 |



# 🔧 Technical Implementation

    Data Preprocessing: Implemented a tailored pipeline for each dataset including StandardScaler for amount features, one-hot encoding for categorical variables (type), and feature engineering (errorBalanceOrig, errorBalanceDest).

    Model Training: Handled severe class imbalance using scale_pos_weight for XGBoost and class_weight='balanced' for scikit-learn models. Utilised stratified sampling in all train-test splits.

    Explainability: Integrated SHAP TreeExplainer to generate explanations. SHAP values for ensemble models are derived from the dominant model component (Random Forest).

    Performance Optimisation: Leveraged parallel processing with n_jobs=-1 where possible and used Streamlit's caching (@st.cache_data, @st.cache_resource) to ensure a smooth user experience by preventing redundant computations and model loading.


# 🎓 Research Contributions

    Comprehensive Model Comparison: Systematic evaluation of multiple ML approaches, including advanced stacking, with statistical significance testing between models.

    Production-Ready Implementation: Real-time processing capabilities, a scalable architecture, and a user-friendly interface for non-technical stakeholders.

    Explainable AI Integration: SHAP-based explanations for regulatory compliance, business impact visualisation for decision-makers, and feature importance analysis for model insights.

    Business Value Assessment: Cost-benefit analysis of fraud detection, false alarm rate optimisation, and ROI calculations for model deployment.


# 📊 Results
Key Findings

    - Ensemble methods generally provide more stable performance, excelling on the complex PaySim dataset.

    - XGBoost is a top performer on structured financial data, demonstrating that model selection is highly dataset-dependent.

    - SHAP explanations are critical for regulatory compliance and enable operational teams to trust and act on model outputs.

    - Business impact analysis reveals that even statistically accurate models can be financially unviable without careful threshold tuning to manage false alarms.


# 🔮 Future Work

    Technical Enhancements:

        - Explore Deep Learning models (e.g., Autoencoders) for anomaly detection.

        - Implement online learning to allow models to adapt to new fraud patterns in real-time.

        - Investigate federated learning to train models across institutions without sharing sensitive data.

    System Improvements:

        - Develop a REST API for programmatic access to the models.

        -  Containerise the application with Docker for easy deployment.

        - Implement an automated retraining and deployment pipeline (CI/CD).



# 🙏 Acknowledgments

    Sincere gratitude to the University of Gloucestershire for their support and guidance.

    Thank you to the Kaggle community for providing the high-quality datasets used in this research.

    This project was made possible by the incredible open-source communities behind scikit-learn, XGBoost, SHAP, and Streamlit.SH


# 📞 Contact
Brandon Elward

- 📧 Email: [brandonelward@gmail.com]
- LinkedIn: [https://www.linkedin.com/in/brandon-elward/]