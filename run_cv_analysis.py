import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                           make_scorer, precision_score, recall_score,
                           roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set a professional plotting style for visualisations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class FraudDetectionCrossValidation:
    """
    A class to encapsulate the entire cross-validation and model evaluation
    process for a fraud detection project.
    """

    def __init__(self, random_state=42, results_directory='results/fraud_detection_analysis'):
        """
        Initialises the analyser.
        """
        self.random_state = random_state
        self.results_dir = Path(results_directory)
        self.cv_results_store = {}
        # Create directories if they don't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {self.results_dir.resolve()}")

    def preprocess_credit_card_data(self, df):
        """Prepares the credit card dataset for model training."""
        df_processed = df.copy()
        scaler = StandardScaler()
        df_processed['scaled_Amount'] = scaler.fit_transform(
            df_processed['Amount'].values.reshape(-1, 1)
        )
        df_processed = df_processed.drop(['Time', 'Amount'], axis=1)
        X = df_processed.drop('Class', axis=1)
        y = df_processed['Class']
        return X, y

    def preprocess_transactions_data(self, df):
        """Prepares the transactions dataset for model training."""
        df_processed = df.copy()
        columns_to_drop = ['step', 'nameOrig', 'nameDest', 'isFlaggedFraud']
        df_processed = df_processed.drop(
            columns=[col for col in columns_to_drop if col in df_processed.columns]
        )
        if 'type' in df_processed.columns:
            type_dummies = pd.get_dummies(df_processed['type'], prefix='type', drop_first=True)
            df_processed = pd.concat([df_processed, type_dummies], axis=1)
            df_processed = df_processed.drop('type', axis=1)
        df_processed['errorBalanceOrig'] = (
            df_processed['oldbalanceOrg'] - 
            df_processed['amount'] - 
            df_processed['newbalanceOrig']
        )
        df_processed['errorBalanceDest'] = (
            df_processed['oldbalanceDest'] + 
            df_processed['amount'] - 
            df_processed['newbalanceDest']
        )
        X = df_processed.drop('isFraud', axis=1)
        y = df_processed['isFraud']
        return X, y

    def get_models(self, y_train):
        """Returns a dictionary of machine learning models to be evaluated."""
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state, class_weight='balanced', max_iter=1000, solver='liblinear'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, class_weight='balanced', n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                random_state=self.random_state, scale_pos_weight=scale_pos_weight, 
                use_label_encoder=False, eval_metric='logloss', n_jobs=-1
            )
        }
        
        ensemble_clf = VotingClassifier(
            estimators=[('rf', models['Random Forest']), ('lr', models['Logistic Regression'])],
            voting='soft', weights=[0.7, 0.3]
        )
        models['Ensemble (RF+LR)'] = ensemble_clf
        return models

    def get_scoring_metrics(self):
        """Defines the scoring metrics for model evaluation."""
        return {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score, zero_division=0),
            'f1': make_scorer(f1_score, zero_division=0),
            'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
            'average_precision': make_scorer(average_precision_score, needs_proba=True)
        }

    def perform_cross_validation(self, X, y, models, cv_folds=5, dataset_name='Dataset'):
        """Performs stratified cross-validation on multiple models."""
        print(f"\n{'='*80}\nPERFORMING CROSS-VALIDATION: {dataset_name}\n{'='*80}")
        print(f"Dataset shape: {X.shape}\nFraud cases: {y.sum():,} ({y.mean()*100:.2f}%)\nCV folds: {cv_folds}")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        scoring = self.get_scoring_metrics()
        results = {}
        
        for model_name, model in models.items():
            print(f"\n--> Evaluating {model_name}...")
            try:
                cv_run_results = cross_validate(estimator=model, X=X, y=y, cv=skf, scoring=scoring, n_jobs=-1)
                results[model_name] = cv_run_results
                for metric_name, scorer in scoring.items():
                    mean_score = cv_run_results[f'test_{metric_name}'].mean()
                    std_score = cv_run_results[f'test_{metric_name}'].std()
                    print(f"  - {metric_name.replace('_', ' ').title():<20}: {mean_score:.4f} ± {std_score:.4f}")
            except Exception as e:
                print(f"  - Error evaluating {model_name}: {e}")
        
        self.cv_results_store[dataset_name] = results
        return results

    def create_summary_table(self, results, dataset_name):
        """Creates and saves a formatted summary table of CV results."""
        summary_data = []
        metrics = self.get_scoring_metrics().keys()
        
        for model_name, model_results in results.items():
            row = {'Model': model_name}
            for metric in metrics:
                mean_val = model_results[f'test_{metric}'].mean()
                std_val = model_results[f'test_{metric}'].std()
                row[metric.replace('_', ' ').title()] = f"{mean_val:.4f} ± {std_val:.4f}"
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        save_path = self.results_dir / f'cv_summary_{dataset_name.lower().replace(" ", "_")}.csv'
        summary_df.to_csv(save_path, index=False)
        print(f"\nSummary table saved to: {save_path}")
        return summary_df

    def plot_cv_results(self, results, dataset_name):
        """Creates and saves bar chart visualisations of CV results."""
        metrics = ['test_f1', 'test_average_precision', 'test_recall', 'test_precision']
        metric_names = ['F1-Score', 'Average Precision (AUPRC)', 'Recall', 'Precision']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(f'Cross-Validation Performance Comparison: {dataset_name}', fontsize=16, fontweight='bold')
        axes = axes.ravel()

        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx]
            model_names = list(results.keys())
            means = [results[model][metric].mean() for model in model_names]
            stds = [results[model][metric].std() for model in model_names]
            
            bars = ax.bar(model_names, means, yerr=stds, capsize=5, alpha=0.8, color=sns.color_palette('viridis', len(model_names)))
            ax.set_ylabel(name, fontsize=12)
            ax.set_title(f'{name} by Model', fontsize=14)
            ax.tick_params(axis='x', rotation=45, labelsize=10)
            
            for bar, mean_val, std_val in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height + (std_val * 1.1),
                        f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = self.results_dir / f'cv_performance_barchart_{dataset_name.lower().replace(" ", "_")}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPerformance bar chart saved to: {save_path}")
        plt.close(fig) # Close the plot to free up memory

    def plot_metric_distributions(self, results, dataset_name):
        """Creates and saves box plots of metric distributions."""
        metrics_to_plot = ['test_f1', 'test_average_precision']
        metric_names = ['F1-Score', 'Average Precision (AUPRC)']
        
        plot_data = []
        for model, res in results.items():
            for metric, name in zip(metrics_to_plot, metric_names):
                for score in res[metric]:
                    plot_data.append({'Model': model, 'Metric': name, 'Score': score})
        
        plot_df = pd.DataFrame(plot_data)
        
        plt.figure(figsize=(12, 7))
        sns.boxplot(x='Metric', y='Score', hue='Model', data=plot_df)
        plt.title(f'Metric Distributions Across Folds: {dataset_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Metric', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.legend(title='Model')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        save_path = self.results_dir / f'cv_metric_distribution_{dataset_name.lower().replace(" ", "_")}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nMetric distribution plot saved to: {save_path}")
        plt.close()

    def save_full_results(self):
        """Saves the detailed results dictionary to a file."""
        save_path = self.results_dir / 'cross_validation_full_results.joblib'
        joblib.dump(self.cv_results_store, save_path)
        print(f"\nDetailed CV results object saved to: {save_path}")

def main():
    """Main function to orchestrate the fraud detection analysis."""
    data_dir = Path('./data') # Relative to the script location
    results_dir = Path('./results/fraud_detection_analysis')
    cv_analyser = FraudDetectionCrossValidation(random_state=42, results_directory=results_dir)
    
    # Credit Card Dataset Analysis
    print("\n" + "="*80 + "\nSTARTING CREDIT CARD DATASET ANALYSIS\n" + "="*80)
    try:
        df_creditcard = pd.read_csv(data_dir / 'creditcard.csv')
        X_cc, y_cc = cv_analyser.preprocess_credit_card_data(df_creditcard)
        models_cc = cv_analyser.get_models(y_cc)
        results_cc = cv_analyser.perform_cross_validation(X_cc, y_cc, models_cc, dataset_name='Credit Card')
        cv_analyser.create_summary_table(results_cc, 'Credit Card')
        cv_analyser.plot_cv_results(results_cc, 'Credit Card')
        cv_analyser.plot_metric_distributions(results_cc, 'Credit Card')
    except FileNotFoundError:
        print(f"\n[WARNING] Credit card dataset not found at '{data_dir / 'creditcard.csv'}'. Skipping.")
    
    # Transactions Dataset Analysis
    print("\n" + "="*80 + "\nSTARTING TRANSACTIONS DATASET ANALYSIS\n" + "="*80)
    try:
        df_transactions = pd.read_csv(data_dir / 'transactions_sample_280k.csv')
        X_trans, y_trans = cv_analyser.preprocess_transactions_data(df_transactions)
        models_trans = cv_analyser.get_models(y_trans)
        results_trans = cv_analyser.perform_cross_validation(X_trans, y_trans, models_trans, dataset_name='Transactions')
        cv_analyser.create_summary_table(results_trans, 'Transactions')
        cv_analyser.plot_cv_results(results_trans, 'Transactions')
        cv_analyser.plot_metric_distributions(results_trans, 'Transactions')
    except FileNotFoundError:
        print(f"\n[WARNING] Transactions dataset not found at '{data_dir / 'transactions_sample_280k.csv'}'. Skipping.")

    cv_analyser.save_full_results()
    print("\n" + "="*80 + "\nANALYSIS COMPLETE\n" + "="*80)

if __name__ == "__main__":
    main()
