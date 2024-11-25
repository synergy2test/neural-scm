import pandas as pd
import numpy as np
from model import DemandPredictor
from data_generator import SupplyChainDataGenerator
from model_explainer import ModelExplainer
import matplotlib.pyplot as plt

def run_scenario_with_explanations(model, data, scenario_name, changes=None):
    """Run a scenario and generate detailed explanations."""
    print(f"\nAnalyzing Scenario: {scenario_name}")
    print("=" * 50)
    
    # Prepare base features
    X, y = model.prepare_data(data)
    
    # Apply scenario changes if provided
    if changes:
        scenario_data = data.copy()
        for key, value in changes.items():
            if key in scenario_data.columns:
                scenario_data[key] *= value
        X_scenario, _ = model.prepare_data(scenario_data)
    else:
        X_scenario = X
    
    # Get predictions
    predictions = model.predict(X_scenario)
    # Ensure predictions are positive
    predictions = np.maximum(0, predictions)
    
    # Get SHAP explanations
    explainer = ModelExplainer(model, model.feature_names)
    shap_values = explainer.get_shap_values(X_scenario[:100])  # Analyze first 100 samples
    
    # Scale SHAP values by feature standard deviation for better interpretability
    feature_std = X_scenario.std()
    scaled_shap_values = shap_values * feature_std.values.reshape(1, -1)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    explainer.plot_feature_importance(scaled_shap_values, X_scenario[:100])
    plt.title(f"Feature Importance - {scenario_name}")
    plt.show()
    
    # Get detailed explanation for a sample prediction
    sample_idx = 0  # Explain first prediction
    explanation = explainer.explain_prediction(X_scenario[sample_idx], scaled_shap_values[sample_idx])
    
    # Print results
    print("\nScenario Results:")
    print(f"Average Predicted Demand: {predictions.mean():.2f}")
    print(f"Demand Range: {predictions.min():.2f} - {predictions.max():.2f}")
    
    print("\nTop Contributing Factors:")
    for factor in explanation['top_factors']:
        impact = factor['contribution']
        direction = "increased" if impact > 0 else "decreased"
        # Scale the impact for reporting
        scaled_impact = impact * feature_std[factor['feature']]
        print(f"- {factor['feature']}: {direction} demand by {abs(scaled_impact):.2f} units")
    
    # Get LLM-based explanations
    try:
        report = explainer.generate_explanation_report(X_scenario[:10], predictions[:10])
        print("\nAI Analysis:")
        if "openai" in report["explanations"]:
            print("\nOpenAI Insights:")
            print(report["explanations"]["openai"])
        print("\nOllama Insights:")
        print(report["explanations"]["ollama"])
    except Exception as e:
        print(f"\nNote: LLM analysis unavailable - {str(e)}")
    
    return {
        'predictions': predictions,
        'shap_values': scaled_shap_values,
        'explanation': explanation,
        'report': report if 'report' in locals() else None
    }

def main():
    # Generate sample data
    print("Generating supply chain data...")
    data_gen = SupplyChainDataGenerator()
    base_data = data_gen.generate_demand_data(num_days=100)
    
    # Create and train model
    print("\nTraining demand prediction model...")
    model = DemandPredictor()
    model.train(base_data, epochs=50)
    
    # Run base scenario
    base_results = run_scenario_with_explanations(
        model, 
        base_data,
        "Base Scenario"
    )
    
    # Run high demand scenario
    high_demand_changes = {
        'price': 0.9,  # 10% price reduction
        'promotion_active': 1.5,  # 50% more promotions
        'inventory_level': 1.3  # 30% more inventory
    }
    high_demand_results = run_scenario_with_explanations(
        model,
        base_data,
        "High Demand Scenario",
        high_demand_changes
    )
    
    # Run supply chain disruption scenario
    disruption_changes = {
        'lead_time_days': 1.5,  # 50% longer lead times
        'inventory_level': 0.7,  # 30% less inventory
        'fulfillment_rate': 0.8  # 20% lower fulfillment rate
    }
    disruption_results = run_scenario_with_explanations(
        model,
        base_data,
        "Supply Chain Disruption Scenario",
        disruption_changes
    )
    
    print("\nScenario Analysis Complete!")
    print("Check the plots above for feature importance visualizations.")

if __name__ == "__main__":
    main()
