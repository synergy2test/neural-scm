import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from datetime import datetime, timedelta
from data_generator import SupplyChainDataGenerator
from model import DemandPredictor
from explainer import AIExplainer
import os
import json

def create_scenario_interface():
    with gr.Blocks() as app:
        gr.Markdown("# 🏭 Supply Chain Scenario Planner")
        
        # Function to list available scenarios
        def list_scenarios():
            try:
                if not os.path.exists("scenarios"):
                    return []
                scenarios = [f.replace(".json", "").replace("_", " ").title() 
                           for f in os.listdir("scenarios") 
                           if f.endswith(".json")]
                return ["New Scenario"] + sorted(scenarios)
            except:
                return ["New Scenario"]
        
        # Function to load scenario
        def load_scenario(scenario_name):
            if scenario_name == "New Scenario":
                return [
                    "Base Scenario",                    # name
                    datetime.now().strftime('%Y-%m-%d'), # date
                    0,                                  # price
                    0,                                  # promo
                    "None",                            # competitor
                    0,                                 # inventory
                    0,                                 # lead time
                    95,                                # fulfillment
                    "Normal",                          # seasonality
                    "None",                            # disruption
                    0,                                 # growth
                    "OpenAI"                           # ai model
                ]
            
            try:
                filename = f"scenarios/{scenario_name.lower().replace(' ', '_')}.json"
                with open(filename, "r") as f:
                    data = json.load(f)
                return [
                    data["name"],
                    data["date"],
                    data["price_change"],
                    data["promotion_intensity"],
                    data["competitor_action"],
                    data["inventory_level"],
                    data["lead_time"],
                    data["fulfillment_target"],
                    data["seasonality"],
                    data["disruption"],
                    data["market_growth"],
                    data["ai_model"]
                ]
            except Exception as e:
                print(f"Error loading scenario: {str(e)}")
                return None
        
        with gr.Row():
            with gr.Column():
                scenario_selector = gr.Dropdown(
                    choices=list_scenarios(),
                    label="Load Scenario",
                    value="New Scenario"
                )
                
                scenario_name = gr.Textbox(
                    label="Scenario Name",
                    placeholder="Enter a name for this scenario",
                    value="Base Scenario"
                )
                
                base_date = gr.Textbox(
                    label="Base Date",
                    value=datetime.now().strftime('%Y-%m-%d'),
                    placeholder="YYYY-MM-DD"
                )
                
                with gr.Column():
                    gr.Markdown("### 🎯 Market Conditions")
                    price_change = gr.Slider(
                        label="Price Change (%)",
                        minimum=-50,
                        maximum=50,
                        value=0,
                        step=1
                    )
                    promotion_intensity = gr.Slider(
                        label="Promotion Intensity (%)",
                        minimum=0,
                        maximum=100,
                        value=0,
                        step=5
                    )
                
                with gr.Column():
                    gr.Markdown("### 📦 Supply Chain Parameters")
                    inventory_level = gr.Slider(
                        label="Inventory Level Change (%)",
                        minimum=-50,
                        maximum=100,
                        value=0,
                        step=5
                    )
                    lead_time = gr.Slider(
                        label="Lead Time Change (%)",
                        minimum=-50,
                        maximum=100,
                        value=0,
                        step=5
                    )
                    fulfillment_target = gr.Slider(
                        label="Fulfillment Target (%)",
                        minimum=80,
                        maximum=100,
                        value=95,
                        step=1
                    )
                
                with gr.Column():
                    gr.Markdown("### 🌍 External Factors")
                    competitor_action = gr.Radio(
                        label="Competitor Action",
                        choices=["None", "Price War", "New Entry"],
                        value="None"
                    )
                    seasonality = gr.Radio(
                        label="Seasonality",
                        choices=["Normal", "High Season", "Low Season"],
                        value="Normal"
                    )
                    disruption = gr.Radio(
                        label="Supply Chain Disruption",
                        choices=["None", "Minor", "Major"],
                        value="None"
                    )
                    market_growth = gr.Slider(
                        label="Market Growth (%)",
                        minimum=-20,
                        maximum=50,
                        value=0,
                        step=5
                    )
                
                with gr.Column():
                    gr.Markdown("### 🤖 AI Configuration")
                    ai_model = gr.Radio(
                        label="AI Model for Insights",
                        choices=["OpenAI", "None"],
                        value="OpenAI"
                    )
                
                run_btn = gr.Button("🚀 Run Analysis", variant="primary")
                save_btn = gr.Button("💾 Save Scenario")
                
                status_text = gr.Markdown("Ready to analyze scenario", visible=True)
            
            # Right column for results
            with gr.Column():
                with gr.Column():
                    metrics_display = gr.Markdown(label="Key Metrics")
                    risk_analysis = gr.Markdown(label="Risk Assessment")
                
                comparison_plot = gr.Plot(label="Demand Comparison")
                feature_importance_plot = gr.Plot(label="Feature Importance")
        
        # Full width AI analysis
        with gr.Row():
            with gr.Column(scale=12):
                gr.Markdown("### 🤖 AI Analysis")
                insights = gr.Markdown()
        
        # Add scenario saving functionality
        def save_scenario(name, date, price, promo, competitor, inventory, lead_time, 
                         fulfillment, seasonality, disruption, growth, ai_model):
            try:
                scenario_data = {
                    "name": name,
                    "date": date,
                    "price_change": price,
                    "promotion_intensity": promo,
                    "competitor_action": competitor,
                    "inventory_level": inventory,
                    "lead_time": lead_time,
                    "fulfillment_target": fulfillment,
                    "seasonality": seasonality,
                    "disruption": disruption,
                    "market_growth": growth,
                    "ai_model": ai_model
                }
                
                # Create scenarios directory if it doesn't exist
                os.makedirs("scenarios", exist_ok=True)
                
                # Save scenario to JSON file
                filename = f"scenarios/{name.replace(' ', '_').lower()}.json"
                with open(filename, "w") as f:
                    json.dump(scenario_data, f, indent=2)
                
                return f"✅ Scenario saved as {filename}"
            except Exception as e:
                return f"❌ Error saving scenario: {str(e)}"

        def update_status(stage):
            return f"🔄 {stage}..."

        def run_analysis(*args):
            try:
                # Run the actual analysis
                status, metrics, comp_plot, feat_plot, insight_text, risks = run_scenario_analysis(*args)
                return status, metrics, comp_plot, feat_plot, insight_text, risks
            except Exception as e:
                empty_plot = Figure()
                empty_plot.update_layout(title="Error occurred")
                return (f"❌ Error: {str(e)}", "Error calculating metrics", 
                        empty_plot, empty_plot, "Error generating insights", 
                        "Error calculating risks")

        run_btn.click(
            run_analysis,
            inputs=[
                scenario_name, base_date, price_change, promotion_intensity, 
                competitor_action, inventory_level, lead_time, fulfillment_target,
                seasonality, disruption, market_growth, ai_model
            ],
            outputs=[
                status_text,
                metrics_display, comparison_plot, feature_importance_plot, 
                insights, risk_analysis
            ]
        )

        save_btn.click(
            save_scenario,
            inputs=[
                scenario_name, base_date, price_change, promotion_intensity, 
                competitor_action, inventory_level, lead_time, fulfillment_target,
                seasonality, disruption, market_growth, ai_model
            ],
            outputs=[status_text]
        )

        scenario_selector.change(
            load_scenario,
            inputs=[scenario_selector],
            outputs=[
                scenario_name, base_date, price_change, promotion_intensity, 
                competitor_action, inventory_level, lead_time, fulfillment_target,
                seasonality, disruption, market_growth, ai_model
            ]
        )

    return app

def run_scenario_analysis(name, date, price, promo, competitor, inventory, 
                         lead_time, fulfillment, seasonality, disruption, growth, ai_model):
    try:
        # Early inventory validation
        if inventory <= 0:
            raise ValueError("⚠️ Warning: No available inventory specified. This will result in 100% stockout risk. Please set a positive inventory level.")
            
        # Create scenario configuration
        scenario_config = {
            'price_multiplier': 1 + (price / 100),
            'promotion_multiplier': promo / 100,
            'inventory_multiplier': inventory / 100,
            'lead_time_multiplier': lead_time / 100,
            'market_growth': 1 + (growth / 100),
            'disruption_impact': {'None': 1.0, 'Minor': 0.9, 'Major': 0.7}[disruption],
            'seasonality_factor': {'Normal': 1.0, 'High Season': 1.3, 'Low Season': 0.7}[seasonality],
            'competitor_impact': {'None': 1.0, 'Price War': 0.85, 'New Entry': 0.9}[competitor]
        }
        
        print("Generating data...")
        # Generate baseline and scenario data
        try:
            # Convert the date to a string format if it's a Timestamp
            if isinstance(date, pd.Timestamp):
                date_str = date.strftime('%Y-%m-%d')
            else:
                date_str = str(date)
            
            start_date = pd.to_datetime(date_str)
            print(f"Using start date: {start_date}")
        except Exception as e:
            start_date = pd.Timestamp.now()
            print(f"Invalid date format ({str(e)}), using current date: {start_date}")
            
        generator = SupplyChainDataGenerator(start_date=start_date.strftime('%Y-%m-%d'))
        baseline_data = generator.generate_demand_data(num_days=90)
        
        # Ensure date column is properly formatted
        if 'date' not in baseline_data.columns:
            baseline_data['date'] = pd.date_range(start=start_date, periods=len(baseline_data), freq='D')
        
        scenario_data = apply_scenario_changes(baseline_data, scenario_config)
        
        print("Training model...")
        # Run model predictions
        model = DemandPredictor()
        try:
            history = model.train(baseline_data)
            print("Model training completed")
        except Exception as e:
            print(f"Model training error: {str(e)}")
            raise
        
        print("Making predictions...")
        # Get predictions and explanations
        scenario_predictions = model.predict(scenario_data)
        scenario_predictions = np.maximum(0, scenario_predictions)  # Ensure positive predictions
        
        print("Generating results...")
        # Generate results
        return generate_scenario_results(name, baseline_data, scenario_data, scenario_predictions, scenario_config, ai_model)
    except Exception as e:
        print(f"Error in run_scenario_analysis: {str(e)}")
        empty_fig = Figure()
        empty_fig.update_layout(
            title="No data available",
            annotations=[{
                "text": str(e),
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 14}
            }]
        )
        return (
            {"error": str(e)},
            empty_fig,
            empty_fig,
            f"⚠️ Error occurred while running scenario analysis: {str(e)}",
            {"error": str(e)}
        )

def apply_scenario_changes(data, config):
    scenario_data = data.copy()
    
    # Store baseline values before applying changes
    scenario_data['baseline_price'] = scenario_data['price'].copy()
    scenario_data['baseline_demand'] = scenario_data['demand'].copy()
    
    # Apply multipliers
    scenario_data['price'] *= config['price_multiplier']
    scenario_data['inventory_level'] *= config['inventory_multiplier']
    scenario_data['lead_time_days'] *= config['lead_time_multiplier']
    
    # Apply external factors
    base_impact = (config['market_growth'] * 
                  config['seasonality_factor'] * 
                  config['disruption_impact'] *
                  config['competitor_impact'])
    
    scenario_data['demand'] *= base_impact
    
    return scenario_data

def generate_scenario_results(name, baseline, scenario, predictions, config, ai_model):
    try:
        # Calculate non-risk metrics first
        metrics = {
            "scenario_name": name,
            "average_demand": float(predictions.mean()),
            "demand_change": float((predictions.mean() - baseline['demand'].mean()) / baseline['demand'].mean() * 100),
            "peak_demand": float(predictions.max()),
            "min_demand": float(predictions.min()),
            "fulfillment_risk": calculate_fulfillment_risk(scenario, predictions)
        }
        
        # Create comparison plot
        comparison_plot = create_comparison_plot(baseline, scenario, predictions)
        
        # Calculate feature importance
        scenario_numeric = scenario.copy()
        date_col = None
        if 'date' in scenario_numeric.columns:
            date_col = scenario_numeric['date'].copy()
            scenario_numeric['date'] = pd.to_numeric(pd.to_datetime(scenario_numeric['date']))
        
        feature_importance = calculate_feature_importance(scenario_numeric)
        importance_plot = create_feature_importance_plot(feature_importance)
        
        # Get AI insights
        insights = "Calculating insights..."
        if ai_model != "None":
            explainer = AIExplainer()
            analysis_report = explainer.generate_explanation_report(
                scenario_numeric, 
                predictions, 
                feature_importance,
                use_openai=(ai_model == "OpenAI"),
                use_ollama=(ai_model == "Ollama")
            )
            insights = format_ai_insights(analysis_report)
        else:
            insights = "AI insights disabled"
        
        # Calculate risks
        print("Calculating risks...")
        try:
            risks = {
                "inventory_stockout_risk": calculate_stockout_risk(scenario, predictions),
                "lead_time_risk": assess_lead_time_risk(scenario),
                "price_sensitivity": calculate_price_sensitivity(scenario, predictions)
            }
            risks["overall_risk_score"] = calculate_overall_risk(scenario, predictions, config)
            risk_text = format_risks(risks)
        except Exception as e:
            print(f"Warning: Risk calculation failed - {str(e)}")
            risk_text = "Risk calculation failed: " + str(e)
        
        # Return all results
        return (
            "Analysis complete",  # status
            format_metrics(metrics),  # metrics
            comparison_plot,  # plot
            importance_plot,  # importance
            insights,  # insights
            risk_text  # risks
        )
            
    except Exception as e:
        print(f"Error in generate_scenario_results: {str(e)}")
        empty_fig = Figure()
        empty_fig.update_layout(title="Error occurred")
        return (
            f"Error: {str(e)}",  # status
            "Error calculating metrics",  # metrics
            empty_fig,  # plot
            empty_fig,  # importance
            "Error generating insights",  # insights
            "Error calculating risks"  # risks
        )

def create_comparison_plot(baseline, scenario, predictions):
    fig = Figure()
    
    # Ensure we have a date index and convert to string format
    if 'date' in baseline.columns:
        x_dates = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in baseline['date']]
    else:
        dates = pd.date_range(start='2023-01-01', periods=len(baseline), freq='D')
        x_dates = dates.strftime('%Y-%m-%d').tolist()
    
    # Convert predictions to a list to ensure consistent data type
    y_predictions = predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
    y_baseline = baseline['demand'].tolist()
    
    # Add baseline demand
    fig.add_trace(go.Scatter(
        x=x_dates,
        y=y_baseline,
        name='Baseline Demand',
        line=dict(color='blue', dash='dash')
    ))
    
    # Add predicted demand
    fig.add_trace(go.Scatter(
        x=x_dates,
        y=y_predictions,
        name='Predicted Demand',
        line=dict(color='green')
    ))
    
    fig.update_layout(
        title='Demand Comparison',
        xaxis_title='Date',
        yaxis_title='Demand',
        template='plotly_white'
    )
    
    return fig

def create_feature_importance_plot(feature_importance):
    fig = Figure()
    
    # Convert values to Python floats to avoid numpy dtype issues
    sorted_features = sorted(
        [(k, float(v)) for k, v in feature_importance.items()],
        key=lambda x: abs(x[1]),
        reverse=True
    )
    features, importance = zip(*sorted_features)
    
    fig.add_trace(go.Bar(
        x=list(features),  # Convert to list to ensure consistent type
        y=list(importance),  # Convert to list to ensure consistent type
        marker_color=['green' if i > 0 else 'red' for i in importance]
    ))
    
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Features',
        yaxis_title='Impact on Demand',
        template='plotly_white'
    )
    
    return fig

def format_ai_insights(report):
    if not report or 'explanations' not in report:
        return "No AI insights available"
    
    insights = "## 🤖 AI Analysis\n\n"
    
    if 'openai' in report['explanations']:
        insights += "### OpenAI Insights\n" + report['explanations']['openai'] + "\n\n"
    
    if 'ollama' in report['explanations']:
        insights += "### Local AI Insights\n" + report['explanations']['ollama']
    
    return insights

# Helper functions for risk calculations
def calculate_fulfillment_risk(scenario, predictions):
    """Calculate fulfillment risk considering partial fulfillment capability"""
    try:
        if 'inventory_level' not in scenario.columns:
            print("Warning: inventory_level column not found in scenario data")
            return 0.0
            
        # Calculate the ratio of demand that can be fulfilled
        inventory = np.maximum(0, scenario['inventory_level'])  # Ensure non-negative
        demand = np.maximum(0, predictions)  # Ensure non-negative
        fulfillment_ratio = np.minimum(1, inventory / np.where(demand > 0, demand, 1))
        
        # Risk is inverse of fulfillment capability
        risk = (1 - np.mean(fulfillment_ratio)) * 100
        return min(100, max(0, float(risk)))
    except Exception as e:
        print(f"Error calculating fulfillment risk: {str(e)}")
        return 0.0

def calculate_stockout_risk(scenario, predictions):
    """Calculate stockout risk considering inventory coverage and magnitude of shortfall"""
    try:
        if 'inventory_level' not in scenario.columns:
            print("⚠️ Critical: inventory_level column not found in scenario data")
            return 100.0
            
        inventory = np.maximum(0, scenario['inventory_level'])
        if np.all(inventory == 0):
            print("⚠️ Critical: Zero inventory levels detected across all time periods. This indicates a severe supply chain risk.")
            return 100.0
            
        demand = np.maximum(0, predictions)
        
        # Calculate days of inventory coverage
        coverage_ratio = inventory / np.where(demand > 0, demand, 1)
        
        # Calculate magnitude of potential shortfall
        shortfall = np.maximum(0, demand - inventory)
        shortfall_ratio = np.where(demand > 0, shortfall / demand, 0)
        
        # Combine coverage and shortfall into overall risk
        coverage_risk = np.mean(coverage_ratio < 1) * 60  # Weight: 60%
        shortfall_risk = np.mean(shortfall_ratio) * 40    # Weight: 40%
        
        total_risk = coverage_risk + shortfall_risk
        return min(100, max(0, float(total_risk)))
    except Exception as e:
        print(f"Error calculating stockout risk: {str(e)}")
        return 0.0

def assess_lead_time_risk(scenario):
    """Calculate lead time risk considering industry standards and variability"""
    try:
        if 'lead_time_days' not in scenario.columns:
            print("Warning: lead_time_days column not found in scenario data")
            return 0.0
            
        lead_times = scenario['lead_time_days'].values  # Convert to numpy array
        
        # Consider both absolute magnitude and variability
        mean_lt = np.mean(lead_times)
        std_lt = np.std(lead_times)
        
        # Risk factors:
        # 1. Base risk from mean lead time (longer = higher risk)
        base_risk = min(60, mean_lt / 30 * 40)  # Cap at 60%, assume 30 days is baseline
        
        # 2. Variability risk (higher variability = higher risk)
        variability_risk = min(40, (std_lt / mean_lt) * 40 if mean_lt > 0 else 0)
        
        total_risk = base_risk + variability_risk
        return min(100, max(0, float(total_risk)))
    except Exception as e:
        print(f"Error calculating lead time risk: {str(e)}")
        return 0.0

def calculate_price_sensitivity(scenario, predictions):
    """Calculate price sensitivity as elasticity measure"""
    try:
        if 'price' not in scenario.columns or 'baseline_price' not in scenario.columns:
            print("Warning: price columns not found in scenario data")
            return 0.0
            
        # Calculate percentage changes
        price_change = (scenario['price'] - scenario['baseline_price']) / scenario['baseline_price']
        demand_change = (predictions - scenario['baseline_demand']) / scenario['baseline_demand']
        
        # Calculate price elasticity where price change is non-zero
        mask = price_change != 0
        elasticity = np.where(mask, demand_change / price_change, 0)
        
        # Normalize to 0-1 range and handle outliers
        normalized = np.clip(abs(np.mean(elasticity)), 0, 2) / 2
        return float(normalized)
    except Exception as e:
        print(f"Error calculating price sensitivity: {str(e)}")
        return 0.0

def calculate_overall_risk(scenario, predictions, config):
    """Calculate overall risk using weighted components and proper normalization"""
    try:
        print("Calculating component risks...")
        
        # Calculate component risks
        stockout = calculate_stockout_risk(scenario, predictions)
        print(f"Stockout risk: {stockout}")
        
        lead_time = assess_lead_time_risk(scenario)
        print(f"Lead time risk: {lead_time}")
        
        price = calculate_price_sensitivity(scenario, predictions)
        print(f"Price sensitivity: {price}")
        
        # Weight the components (total = 100):
        # - Stockout: 40% (most critical)
        # - Lead Time: 35%
        # - Price Sensitivity: 25%
        overall = (
            stockout * 0.40 +
            lead_time * 0.35 +
            (price * 100) * 0.25  # Convert price sensitivity to percentage
        )
        
        print(f"Overall risk: {overall}")
        return min(100, max(0, float(overall)))
    except Exception as e:
        print(f"Error calculating overall risk: {str(e)}")
        return 0.0

def calculate_feature_importance(data):
    # Calculate correlations only for numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    importance = {}
    
    for col in numeric_cols:
        if col != 'demand':
            try:
                # Remove any NaN, infinite, or zero values before correlation
                valid_mask = ~(np.isnan(data[col]) | np.isinf(data[col]) | 
                             np.isnan(data['demand']) | np.isinf(data['demand']))
                
                # Ensure we have enough valid data points
                if valid_mask.sum() > 1:  # Need at least 2 points for correlation
                    x = data[col][valid_mask].astype(float)
                    y = data['demand'][valid_mask].astype(float)
                    
                    # Check for zero variance
                    if x.std() != 0 and y.std() != 0:
                        correlation = float(np.corrcoef(x, y)[0, 1])
                        importance[col] = correlation if not np.isnan(correlation) else 0.0
                    else:
                        importance[col] = 0.0
                else:
                    importance[col] = 0.0
            except Exception as e:
                print(f"Error calculating correlation for {col}: {str(e)}")
                importance[col] = 0.0
    
    return importance

def format_metrics(metrics):
    return f"### Key Metrics\n\n" \
           f"* **Scenario Name**: {metrics['scenario_name']}\n" \
           f"* **Average Demand**: {metrics['average_demand']:.2f}\n" \
           f"* **Demand Change**: {metrics['demand_change']:.2f}%\n" \
           f"* **Peak Demand**: {metrics['peak_demand']:.2f}\n" \
           f"* **Min Demand**: {metrics['min_demand']:.2f}\n" \
           f"* **Fulfillment Risk**: {metrics['fulfillment_risk']:.2f}%"

def format_risks(risks):
    return f"""### Risk Assessment
* Inventory Stockout Risk: {risks['inventory_stockout_risk']:.1f}%
* Lead Time Risk: {risks['lead_time_risk']:.1f}%
* Price Sensitivity: {risks['price_sensitivity']:.2f}
* Overall Risk Score: {risks['overall_risk_score']:.1f}%"""

if __name__ == "__main__":
    app = create_scenario_interface()
    app.launch(share=False)
