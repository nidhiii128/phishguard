import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(df, figsize=(12, 10)):
    """Plot correlation matrix of features"""
    plt.figure(figsize=figsize)
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    return plt.gcf()

def plot_feature_distributions(df, features, n_cols=3):
    """Plot distributions of selected features"""
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        if feature in df.columns:
            df[feature].hist(bins=50, ax=axes[i])
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
    
    # Hide empty subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    return fig