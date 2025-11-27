import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

from feature_engineering import create_feature_dataframe

class PhishingURLClassifier:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_and_preprocess_data(self, data_path, sample_size=50000):
        """Load data and create features"""
        print("Loading data...")
        # Resolve relative paths against this file's directory so the script works
        # when invoked from different working directories
        if not os.path.isabs(data_path):
            data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), data_path))

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        df = pd.read_csv(data_path)
        
        # Sample data if too large for quick prototyping
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size} records for training")
        
        print(f"Original data shape: {df.shape}")

        # Try to find the label column in a robust way
        possible_label_names = ['label', 'Label', 'status', 'Status', 'class', 'Class', 'type', 'Type', 'result', 'Result']
        label_col = None
        for name in possible_label_names:
            if name in df.columns:
                label_col = name
                break

        # If not found by name, try heuristic: last column often contains labels
        if label_col is None:
            # check for a column with only a small set of string values like 'phishing'/'legitimate'
            for col in df.columns[-3:]:
                unique_vals = set(df[col].dropna().astype(str).str.lower().unique())
                if unique_vals <= set(['phishing', 'legitimate']) or unique_vals & set(['phishing', 'legitimate']):
                    label_col = col
                    break

        # Fallback: if still not found, use the last column
        if label_col is None:
            label_col = df.columns[-1]

        print(f"Using label column: '{label_col}'")
        try:
            print(f"Label distribution:\n{df[label_col].value_counts()}")
        except Exception:
            print("Could not print label distribution (column may contain unprintable values)")
        
        # Detect URL column name (common variants)
        possible_url_names = ['url', 'URL', 'link', 'Link', 'website', 'Website', 'domain', 'Domain']
        url_col = None
        for name in possible_url_names:
            if name in df.columns:
                url_col = name
                break

        # If not found, try heuristics: first column containing 'http' in sample values
        if url_col is None:
            for col in df.columns[:5]:
                sample_vals = df[col].dropna().astype(str).head(10).tolist()
                if any(v.startswith('http') or v.startswith('https') for v in sample_vals):
                    url_col = col
                    break

        # Fallback to the first column
        if url_col is None:
            url_col = df.columns[0]

        print(f"Using URL column: '{url_col}'")

        # Extract features
        print("Extracting features from URLs...")
        # Pass the detected URL and label column values to the feature extractor so it will attach a 'label' column
        feature_df = create_feature_dataframe(df[url_col], df[label_col])

        # Remove label column for features and create y
        if 'label' in feature_df.columns:
            y_series = feature_df['label']

            # Define mapping for common textual labels
            def map_label_to_binary(val):
                if pd.isna(val):
                    return 0
                try:
                    # If numeric-like (0/1), cast
                    if isinstance(val, (int, float)):
                        return int(val)
                    s = str(val).strip().lower()
                    if s in ['phishing', 'phish', 'malicious', 'bad', '1', 'true', 'yes']:
                        return 1
                    if s in ['legitimate', 'legit', 'good', 'benign', '0', 'false', 'no']:
                        return 0
                    # some datasets use 'suspicious' for phishing
                    if 'phish' in s or 'suspicious' in s or 'malware' in s:
                        return 1
                    if 'legit' in s or 'good' in s or 'benign' in s:
                        return 0
                except Exception:
                    pass
                # As a last resort, try to interpret as int
                try:
                    return int(float(val))
                except Exception:
                    # Default to 0 (non-phishing) for unknown labels
                    return 0

            y = y_series.apply(map_label_to_binary).astype(int)
            X = feature_df.drop('label', axis=1)
        else:
            raise ValueError("Label column not found in feature DataFrame")
        
        # Handle NaN values
        X = X.fillna(0)
        
        self.feature_columns = X.columns.tolist()
        
        print(f"Feature matrix shape: {X.shape}")
        return X, y
    
    def train_models(self, X, y):
        """Train multiple models and compare performance"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models to train
        models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            if name == 'Naive Bayes':
                # Naive Bayes requires scaled data
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                self.models[name] = (model, True)  # Store with scale flag
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                self.models[name] = (model, False)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'predictions': y_pred,
                'scaled': name == 'Naive Bayes'
            }
            
            print(f"{name} Results:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"  Confusion Matrix:\n{cm}")
        
        self.results = results
        self.X_test = X_test
        self.y_test = y_test
        
        return results
    
    def get_best_model(self):
        """Return the best model based on F1 score"""
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['f1'])
        return best_model_name, self.results[best_model_name]
    
    def save_model(self, model_name, model_path='../models/trained_model.joblib'):
        """Save the best model and feature columns"""
        if not os.path.exists('../models'):
            os.makedirs('../models')
            
        best_model_name, best_result = self.get_best_model()
        model, needs_scaling = self.models[best_model_name]
        
        # Save model, scaler, and feature columns
        model_data = {
            'model': model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'needs_scaling': needs_scaling,
            'model_name': best_model_name,
            'performance': best_result
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved as {model_path}")
        
        # Also save feature columns separately for easy access
        joblib.dump(self.feature_columns, '../models/feature_columns.joblib')
        
    def plot_feature_importance(self, top_n=15):
        """Plot feature importance for tree-based models"""
        best_model_name, _ = self.get_best_model()
        
        if best_model_name in ['Decision Tree', 'Random Forest']:
            model, _ = self.models[best_model_name]
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = self.feature_columns
                
                # Create DataFrame for plotting
                feat_imp = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False).head(top_n)
                
                plt.figure(figsize=(10, 8))
                sns.barplot(data=feat_imp, x='importance', y='feature')
                plt.title(f'Top {top_n} Most Important Features - {best_model_name}')
                plt.tight_layout()
                plt.savefig('../models/feature_importance.png', dpi=300, bbox_inches='tight')
                plt.show()

def main():
    classifier = PhishingURLClassifier()
    
    # Load and preprocess data
    X, y = classifier.load_and_preprocess_data('../data/phishing_site_urls.csv')
    
    # Train models
    results = classifier.train_models(X, y)
    
    # Get best model
    best_name, best_result = classifier.get_best_model()
    print(f"\n🎯 Best Model: {best_name}")
    print(f"📊 Best F1-Score: {best_result['f1']:.4f}")
    
    # Save model
    classifier.save_model(best_name)
    
    # Plot feature importance
    classifier.plot_feature_importance()
    
    return classifier

if __name__ == "__main__":
    classifier = main()