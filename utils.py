import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_validate_data(filepath):
    """
    Load and validate the diabetes dataset.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Validated dataframe
    """
    required_columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    
    data = pd.read_csv(filepath)
    
    # Check for required columns
    missing_cols = set(required_columns) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return data


def plot_feature_distributions(data, save_path=None):
    """
    Plot distribution of all features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to plot
    save_path : str, optional
        Path to save the plot
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        axes[idx].hist(data[col], bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribution of {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
    
    # Hide extra subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def calculate_metrics_summary(y_true, y_pred):
    """
    Calculate and return a summary of classification metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
        
    Returns:
    --------
    dict
        Dictionary containing all metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, confusion_matrix
    )
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'specificity': tn / (tn + fp),
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    }
    
    return metrics


def create_prediction_report(metrics, save_path='outputs/metrics_report.txt'):
    """
    Create a formatted report of prediction metrics.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics
    save_path : str
        Path to save the report
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("PREDICTION METRICS REPORT\n")
        f.write("="*60 + "\n\n")
        
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key.capitalize():.<30} {value:.4f}\n")
            else:
                f.write(f"{key.capitalize():.<30} {value}\n")
    
    print(f"Report saved to {save_path}")
