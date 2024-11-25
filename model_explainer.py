"""
Model explainability module for supply chain predictions using LLMs and SHAP.
"""
import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from openai import OpenAI
import requests

class ModelExplainer:
    def __init__(self, model, feature_names):
        """Initialize explainer with model and feature names."""
        self.model = model
        self.feature_names = feature_names
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.ollama_url = "http://localhost:11434/api/generate"
        self.shap_explainer = None
        
    def _initialize_shap(self, background_data):
        """Initialize SHAP explainer with background data."""
        # Use KernelExplainer for model-agnostic explanations
        self.shap_explainer = shap.KernelExplainer(self.model.predict, background_data)
        
    def get_shap_values(self, X):
        """Calculate SHAP values for predictions."""
        if self.shap_explainer is None:
            self._initialize_shap(X[:100])  # Use first 100 samples as background
        return self.shap_explainer.shap_values(X)
        
    def plot_feature_importance(self, shap_values, X, feature_names=None):
        """Plot global feature importance based on SHAP values."""
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, feature_names=feature_names or self.feature_names)
        plt.tight_layout()
        
    def explain_prediction(self, instance, shap_values=None):
        """Generate detailed explanation for a single prediction."""
        if shap_values is None:
            shap_values = self.get_shap_values([instance])[0]
            
        # Get feature contributions
        contributions = {}
        for i, (name, value) in enumerate(zip(self.feature_names, instance)):
            contributions[name] = {
                'value': float(value),
                'impact': float(shap_values[i])
            }
            
        # Sort features by absolute impact
        sorted_features = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]['impact']),
            reverse=True
        )
        
        # Generate natural language explanation
        explanation = {
            'prediction': float(self.model.predict([instance])[0]),
            'feature_contributions': contributions,
            'top_factors': [
                {
                    'feature': feat,
                    'contribution': details['impact'],
                    'value': details['value']
                }
                for feat, details in sorted_features[:3]
            ]
        }
        
        return explanation
    
    def explain_with_openai(self, scenario_data, predictions):
        """Get explanation from OpenAI."""
        scenario_json = self._format_scenario_data(scenario_data, predictions)
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a supply chain analytics expert. Analyze the prediction scenario and provide concise, actionable insights."},
                {"role": "user", "content": f"Analyze this demand prediction scenario and provide key insights:\n{json.dumps(scenario_json, indent=2)}"}
            ],
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def explain_with_ollama(self, scenario_data, predictions, model_name="llama2"):
        """Get explanation from Ollama."""
        scenario_json = self._format_scenario_data(scenario_data, predictions)
        
        prompt = f"""
        As a supply chain expert, analyze this demand prediction scenario:
        {json.dumps(scenario_json, indent=2)}
        
        Provide a brief explanation of:
        1. Key factors driving the predictions
        2. Potential risks or anomalies
        3. Recommendations for supply chain managers
        
        Keep the response concise and focused on actionable insights.
        """
        
        response = requests.post(self.ollama_url, json={
            "model": model_name,
            "prompt": prompt,
            "stream": False
        })
        
        return response.json()["response"]
    
    def _get_feature_importance(self, X, y_pred):
        """Calculate feature importance using permutation."""
        importance = {}
        baseline_pred = y_pred.copy()
        
        for feature in self.feature_names:
            # Permute feature values
            X_permuted = X.copy()
            X_permuted[feature] = np.random.permutation(X_permuted[feature])
            
            # Calculate impact on predictions
            pred_permuted = self.model.predict(X_permuted)
            importance[feature] = np.mean(np.abs(baseline_pred - pred_permuted))
            
        return importance
    
    def _format_scenario_data(self, scenario_data, predictions):
        """Format scenario data for LLM consumption."""
        return {
            "timestamp": datetime.now().isoformat(),
            "features": {
                name: float(scenario_data[name].mean()) 
                for name in self.feature_names
            },
            "predictions": {
                "mean_demand": float(predictions.mean()),
                "total_demand": float(predictions.sum()),
                "demand_std": float(predictions.std())
            },
            "feature_importance": self._get_feature_importance(scenario_data, predictions)
        }
    
    def generate_explanation_report(self, scenario_data, predictions, use_openai=True):
        """Generate a comprehensive explanation report using both LLMs."""
        report = {
            "data": self._format_scenario_data(scenario_data, predictions),
            "explanations": {}
        }
        
        # Get explanations from both models
        if use_openai:
            report["explanations"]["openai"] = self.explain_with_openai(scenario_data, predictions)
        report["explanations"]["ollama"] = self.explain_with_ollama(scenario_data, predictions)
        
        return report
    
    def save_explanation(self, report, filename=None):
        """Save explanation report to file."""
        if filename is None:
            filename = f"explanation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        return filename
