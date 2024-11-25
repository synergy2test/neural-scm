"""
Example script demonstrating the DemandPredictor with AI-powered explanations.
"""
import numpy as np
from model import DemandPredictor

def generate_sample_data(n_samples=1000):
    """Generate sample data for demonstration."""
    np.random.seed(42)
    X = np.random.randn(n_samples, 4)  # 4 features
    y = 2 * X[:, 0] + 0.5 * X[:, 1] - X[:, 2] + 0.1 * X[:, 3] + np.random.randn(n_samples) * 0.1
    return X, y

def main():
    # Generate sample data
    X, y = generate_sample_data()
    
    # Split into train and test sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Initialize model with AI explainer enabled
    predictor = DemandPredictor(use_ai_explainer=True)
    
    # Train the model
    history = predictor.train(X_train, y_train, epochs=20)
    
    # Evaluate and get AI-powered explanations
    results = predictor.evaluate(X_test, y_test)
    
    # Print metrics
    print("\nModel Metrics:")
    print(f"MSE: {results['mse']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"MAE: {results['mae']:.4f}")
    
    # Print sample predictions
    print("\nSample Predictions:")
    for pred in results['predictions']:
        print(f"Actual: {pred['actual']:.2f}, Predicted: {pred['predicted']:.2f}, "
              f"Difference: {pred['difference']:.2f}")
    
    # Print AI explanations
    if 'ai_explanations' in results:
        print("\nOpenAI Explanation:")
        print(results['ai_explanations']['openai_explanation'])
        
        print("\nOllama Explanation:")
        print(results['ai_explanations']['ollama_explanation'])
    elif 'ai_explanations_error' in results:
        print("\nError getting AI explanations:", results['ai_explanations_error'])

if __name__ == "__main__":
    main()
