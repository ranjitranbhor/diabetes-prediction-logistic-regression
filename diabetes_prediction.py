import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import warnings
warnings.filterwarnings('ignore')


class DiabetesPredictor:
    """
    A class to handle diabetes prediction using Logistic Regression.
    """
    
    def __init__(self, data_path='data/diabetes.csv'):
        """
        Initialize the DiabetesPredictor.
        
        Parameters:
        -----------
        data_path : str
            Path to the diabetes dataset CSV file
        """
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.predictions = None
        
    def load_data(self):
        """Load the diabetes dataset from CSV file."""
        print("Loading data...")
        self.data = pd.read_csv(self.data_path)
        print(f"Data loaded successfully. Shape: {self.data.shape}")
        return self.data
    
    def explore_data(self):
        """Perform exploratory data analysis."""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        print("\nFirst 5 observations:")
        print(self.data.head())
        
        print("\nDataset Information:")
        print(self.data.info())
        
        print("\nStatistical Summary:")
        print(self.data.describe())
        
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        
        print("\nTarget Variable Distribution:")
        print(self.data['Outcome'].value_counts())
        print(f"\nClass Distribution (%):")
        print(self.data['Outcome'].value_counts(normalize=True) * 100)
        
    def visualize_data(self, save_plots=False):
        """
        Create visualizations for the dataset.
        
        Parameters:
        -----------
        save_plots : bool
            Whether to save plots to files
        """
        print("\nGenerating visualizations...")
        
        # 1. Missing data heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.data.isnull(), cbar=True, cmap='viridis')
        plt.title('Missing Data Heatmap')
        if save_plots:
            plt.savefig('outputs/missing_data_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Target variable distribution
        plt.figure(figsize=(8, 5))
        sns.countplot(x='Outcome', data=self.data, palette='Set2')
        plt.title('Distribution of Diabetes Outcome')
        plt.xlabel('Outcome (0: No Diabetes, 1: Diabetes)')
        plt.ylabel('Count')
        if save_plots:
            plt.savefig('outputs/outcome_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Correlation heatmap
        plt.figure(figsize=(12, 8))
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlation Heatmap')
        if save_plots:
            plt.savefig('outputs/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations generated successfully!")
        
    def prepare_data(self, test_size=0.3, random_state=468):
        """
        Prepare data for training by splitting into train and test sets.
        
        Parameters:
        -----------
        test_size : float
            Proportion of dataset to include in test split
        random_state : int
            Random seed for reproducibility
        """
        print(f"\nSplitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test...")
        
        # Separate features and target
        X = self.data.drop('Outcome', axis=1)
        y = self.data['Outcome']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        
    def train_model(self, solver='liblinear', random_state=123):
        """
        Train the Logistic Regression model.
        
        Parameters:
        -----------
        solver : str
            Algorithm to use in optimization
        random_state : int
            Random seed for reproducibility
        """
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        
        self.model = LogisticRegression(solver=solver, random_state=random_state)
        self.model.fit(self.X_train, self.y_train)
        
        print("Model trained successfully!")
        print(f"Solver: {solver}")
        print(f"Random State: {random_state}")
        
    def display_coefficients(self):
        """Display model coefficients and their interpretation."""
        print("\n" + "="*60)
        print("MODEL COEFFICIENTS")
        print("="*60)
        
        print(f"\nIntercept: {self.model.intercept_[0]:.6f}")
        
        coef_df = pd.DataFrame({
            'Feature': self.X_train.columns,
            'Coefficient': self.model.coef_[0]
        })
        coef_df = coef_df.sort_values('Coefficient', ascending=False)
        
        print("\nFeature Coefficients (sorted by magnitude):")
        print(coef_df.to_string(index=False))
        
        # Interpretation
        print("\n" + "-"*60)
        print("COEFFICIENT INTERPRETATION:")
        print("-"*60)
        
        strongest_positive = coef_df.iloc[0]
        strongest_negative = coef_df.iloc[-1]
        
        print(f"\nStrongest Positive Predictor: {strongest_positive['Feature']}")
        print(f"  Coefficient: {strongest_positive['Coefficient']:.6f}")
        print(f"  Interpretation: Higher values increase diabetes probability")
        
        if strongest_negative['Coefficient'] < 0:
            print(f"\nStrongest Negative Predictor: {strongest_negative['Feature']}")
            print(f"  Coefficient: {strongest_negative['Coefficient']:.6f}")
            print(f"  Interpretation: Higher values decrease diabetes probability")
        
    def predict(self):
        """Make predictions on the test set."""
        print("\nMaking predictions on test set...")
        self.predictions = self.model.predict(self.X_test)
        print("Predictions completed!")
        
    def evaluate_model(self):
        """Evaluate model performance with multiple metrics."""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, self.predictions)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Diabetes', 'Diabetes'],
                    yticklabels=['No Diabetes', 'Diabetes'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.predictions, 
                                   target_names=['No Diabetes', 'Diabetes']))
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, self.predictions)
        precision = precision_score(self.y_test, self.predictions)
        recall = recall_score(self.y_test, self.predictions)
        f1 = f1_score(self.y_test, self.predictions)
        
        # Calculate Specificity
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        sensitivity = recall  # Same as recall
        
        # Display key metrics
        print("\n" + "-"*60)
        print("KEY PERFORMANCE METRICS")
        print("-"*60)
        print(f"Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision:   {precision:.4f} ({precision*100:.2f}%)")
        print(f"Sensitivity: {sensitivity:.4f} ({sensitivity*100:.2f}%)")
        print(f"Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
        print(f"F1-Score:    {f1:.4f}")
        
        # Model assessment
        print("\n" + "-"*60)
        print("MODEL ASSESSMENT FOR CLINICAL USE")
        print("-"*60)
        
        if sensitivity < 0.70:
            print("⚠️  WARNING: Low sensitivity detected!")
            print(f"   The model misses {(1-sensitivity)*100:.1f}% of diabetes cases.")
            print("   NOT RECOMMENDED for clinical deployment.")
        else:
            print("✓ Sensitivity meets minimum threshold for clinical consideration.")
            
        if specificity < 0.70:
            print("⚠️  WARNING: Low specificity detected!")
            print(f"   The model has {(1-specificity)*100:.1f}% false positive rate.")
        else:
            print("✓ Specificity is acceptable.")
            
        return {
            'accuracy': accuracy,
            'precision': precision,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1_score': f1
        }
    
    def save_results(self, filename='outputs/model_results.txt'):
        """
        Save model results to a text file.
        
        Parameters:
        -----------
        filename : str
            Path to save the results file
        """
        import os
        
        # Create outputs directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            f.write("="*60 + "\n")
            f.write("DIABETES PREDICTION MODEL RESULTS\n")
            f.write("="*60 + "\n\n")
            
            f.write("Model: Logistic Regression\n")
            f.write(f"Dataset: {self.data_path}\n")
            f.write(f"Total Samples: {len(self.data)}\n")
            f.write(f"Training Samples: {len(self.X_train)}\n")
            f.write(f"Test Samples: {len(self.X_test)}\n\n")
            
            f.write("Confusion Matrix:\n")
            cm = confusion_matrix(self.y_test, self.predictions)
            f.write(str(cm) + "\n\n")
            
            f.write("Classification Report:\n")
            f.write(classification_report(self.y_test, self.predictions))
            
        print(f"\nResults saved to {filename}")
    
    def run_full_pipeline(self, save_outputs=False):
        """
        Run the complete analysis pipeline.
        
        Parameters:
        -----------
        save_outputs : bool
            Whether to save plots and results to files
        """
        print("\n" + "="*60)
        print("DIABETES PREDICTION - FULL PIPELINE")
        print("="*60 + "\n")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Explore data
        self.explore_data()
        
        # Step 3: Visualize data
        self.visualize_data(save_plots=save_outputs)
        
        # Step 4: Prepare data
        self.prepare_data()
        
        # Step 5: Train model
        self.train_model()
        
        # Step 6: Display coefficients
        self.display_coefficients()
        
        # Step 7: Make predictions
        self.predict()
        
        # Step 8: Evaluate model
        metrics = self.evaluate_model()
        
        # Step 9: Save results (optional)
        if save_outputs:
            self.save_results()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return metrics


def main():
    """Main function to run the diabetes prediction model."""
    
    # Initialize predictor
    predictor = DiabetesPredictor(data_path='data/diabetes.csv')
    
    # Run full pipeline
    metrics = predictor.run_full_pipeline(save_outputs=True)
    
    # Additional analysis or custom predictions can be added here
    
    return predictor, metrics


if __name__ == "__main__":
    # Run the main function
    predictor, metrics = main()
    
    print("\n" + "="*60)
    print("To use this model for predictions:")
    print("  1. Load the predictor: predictor = DiabetesPredictor()")
    print("  2. Run pipeline: predictor.run_full_pipeline()")
    print("  3. Access model: predictor.model")
    print("  4. Make custom predictions on new data")
    print("="*60)
