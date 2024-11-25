"""
Supply Chain Model Explainer using OpenAI and Ollama
"""
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from openai import OpenAI
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AIExplainer:
    def __init__(self):
        """Initialize the AI explainer with API configurations."""
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'llama2')

    def _format_supply_chain_data(self, data, predictions, feature_importance):
        """Format supply chain data for LLM analysis."""
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "demand_mean": float(predictions.mean()),
                "demand_std": float(predictions.std()),
                "total_demand": float(predictions.sum()),
                "inventory_level_mean": float(data['inventory_level'].mean()),
                "price_mean": float(data['price'].mean()),
                "fulfillment_rate": float(data['fulfillment_rate'].mean()),
            },
            "feature_importance": {
                k: float(v) for k, v in feature_importance.items()
            },
            "trends": {
                "seasonal_pattern": self._detect_seasonality(data),
                "price_elasticity": self._calculate_price_elasticity(data),
                "inventory_efficiency": self._analyze_inventory(data)
            }
        }

    def _detect_seasonality(self, data):
        """Detect seasonal patterns in the data."""
        if 'month' not in data.columns:
            return "No seasonal data available"
        
        monthly_demand = data.groupby('month')['demand'].mean()
        peak_month = monthly_demand.idxmax()
        trough_month = monthly_demand.idxmin()
        seasonality_ratio = monthly_demand.max() / monthly_demand.min()
        
        return {
            "peak_month": int(peak_month),
            "trough_month": int(trough_month),
            "seasonality_strength": float(seasonality_ratio)
        }

    def _calculate_price_elasticity(self, data):
        """Calculate price elasticity of demand."""
        if 'price' not in data.columns or 'demand' not in data.columns:
            return "Price elasticity calculation not available"
        
        price_std = data['price'].std()
        demand_std = data['demand'].std()
        if price_std == 0:
            return 0
        
        correlation = data['price'].corr(data['demand'])
        elasticity = correlation * (demand_std / price_std)
        
        return float(elasticity)

    def _analyze_inventory(self, data):
        """Analyze inventory efficiency metrics."""
        if 'inventory_level' not in data.columns or 'demand' not in data.columns:
            return "Inventory analysis not available"
        
        coverage_ratio = data['inventory_level'].mean() / data['demand'].mean()
        stockout_rate = (data['inventory_level'] < data['demand']).mean()
        
        return {
            "inventory_coverage_days": float(coverage_ratio * 30),  # Assuming monthly data
            "stockout_rate": float(stockout_rate),
            "inventory_turnover": float(data['inventory_turnover'].mean()) if 'inventory_turnover' in data.columns else None
        }

    def explain_with_openai(self, formatted_data):
        """Get supply chain insights from OpenAI."""
        try:
            print("Debug: Preparing OpenAI prompt...")
            system_prompt = """You are an expert supply chain analyst. Analyze the supply chain data and metrics provided, 
            and give actionable insights and recommendations."""
            
            user_prompt = f"""Analyze this supply chain scenario data and provide insights:
            {json.dumps(formatted_data, indent=2)}
            
            Focus on actionable recommendations supported by the data."""
            
            print("Debug: Sending request to OpenAI...")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            print("Debug: Received OpenAI response")
            return response.choices[0].message.content
        except Exception as e:
            print(f"Warning: OpenAI analysis failed - {str(e)}")
            return f"OpenAI analysis unavailable: {str(e)}"

    def explain_with_ollama(self, formatted_data):
        """Get supply chain insights from Ollama."""
        prompt = f"""
        Analyze this supply chain scenario as an expert:
        {json.dumps(formatted_data, indent=2)}
        
        Provide a concise analysis covering:
        1. Key demand drivers and patterns
        2. Inventory recommendations
        3. Pricing insights
        4. Risk assessment
        
        Keep the response focused on practical actions.
        """
        
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            return response.json()["response"]
        except Exception as e:
            print(f"Warning: Ollama analysis failed - {str(e)}")
            return "Ollama analysis unavailable"

    def generate_explanation_report(self, data, predictions, feature_importance, use_openai=True, use_ollama=False):
        """Generate a comprehensive explanation report using selected LLM."""
        try:
            print("Debug: Starting explanation report generation...")
            formatted_data = self._format_supply_chain_data(data, predictions, feature_importance)
            print("Debug: Data formatted successfully")
            
            # Format dates properly for the metadata
            start_date = None
            end_date = None
            if 'date' in data.columns:
                try:
                    start_date = pd.to_datetime(data['date'].min()).isoformat()
                    end_date = pd.to_datetime(data['date'].max()).isoformat()
                except:
                    print("Debug: Could not parse dates for metadata")
            
            report = {
                "data": formatted_data,
                "explanations": {},
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "model_version": "1.0",
                    "data_timeframe": {
                        "start": start_date,
                        "end": end_date
                    }
                }
            }
            
            if use_openai and os.getenv('OPENAI_API_KEY'):
                print("Debug: Using OpenAI for explanation...")
                openai_explanation = self.explain_with_openai(formatted_data)
                if openai_explanation:
                    report["explanations"]["openai"] = openai_explanation
            
            if use_ollama:
                print("Debug: Using Ollama for explanation...")
                ollama_explanation = self.explain_with_ollama(formatted_data)
                if ollama_explanation:
                    report["explanations"]["ollama"] = ollama_explanation
            
            return report
        except Exception as e:
            print(f"Debug: Error in generate_explanation_report - {str(e)}")
            return {
                "data": {},
                "explanations": {},
                "error": str(e)
            }

    def save_explanation(self, report, scenario_name=None):
        """Save the explanation report to a file."""
        if scenario_name is None:
            scenario_name = "supply_chain_analysis"
        
        filename = f"explanations/{scenario_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("explanations", exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        return filename

    def analyze_scenario(self, data, predictions, feature_importance, scenario_name=None):
        """Analyze a specific scenario and generate insights."""
        report = self.generate_explanation_report(data, predictions, feature_importance)
        filename = self.save_explanation(report, scenario_name)
        
        print(f"\nScenario Analysis: {scenario_name or 'Current State'}")
        print("=" * 50)
        
        # Print key metrics
        metrics = report["data"]["metrics"]
        print("\nKey Metrics:")
        print(f"Average Demand: {metrics['demand_mean']:.2f}")
        print(f"Fulfillment Rate: {metrics['fulfillment_rate']*100:.1f}%")
        print(f"Average Inventory: {metrics['inventory_level_mean']:.2f}")
        
        # Print AI insights
        if "openai" in report["explanations"]:
            print("\nOpenAI Analysis:")
            print(report["explanations"]["openai"])
        
        print("\nOllama Analysis:")
        print(report["explanations"]["ollama"])
        
        print(f"\nDetailed report saved to: {filename}")
        return report
