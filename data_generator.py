"""
Supply Chain Data Generator for neural-scm project.
Generates realistic supply chain data for training and testing the demand prediction model.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class SupplyChainDataGenerator:
    def __init__(self, start_date=None, num_products=10, num_locations=5):
        """
        Initialize the supply chain data generator.
        
        Args:
            start_date (str): Starting date for data generation (format: 'YYYY-MM-DD')
            num_products (int): Number of unique products
            num_locations (int): Number of warehouse locations
        """
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d') if start_date else datetime.now()
        self.num_products = num_products
        self.num_locations = num_locations
        
        # Generate product and location metadata
        self.products = self._generate_products()
        self.locations = self._generate_locations()
        
    def _generate_products(self):
        """Generate product metadata with realistic attributes."""
        products = []
        categories = ['Electronics', 'Clothing', 'Food', 'Home Goods', 'Automotive']
        
        for i in range(self.num_products):
            product = {
                'product_id': f'P{i:03d}',
                'category': np.random.choice(categories),
                'base_price': np.random.uniform(10, 1000),
                'base_demand': np.random.uniform(100, 1000),
                'seasonality_factor': np.random.uniform(0.8, 1.2),
                'shelf_life_days': np.random.choice([30, 60, 90, 180, 365]),
            }
            products.append(product)
        
        return pd.DataFrame(products)
    
    def _generate_locations(self):
        """Generate warehouse location metadata."""
        locations = []
        regions = ['North', 'South', 'East', 'West', 'Central']
        
        for i in range(self.num_locations):
            location = {
                'location_id': f'L{i:03d}',
                'region': regions[i % len(regions)],
                'storage_capacity': np.random.uniform(10000, 50000),
                'handling_cost': np.random.uniform(5, 15),
            }
            locations.append(location)
        
        return pd.DataFrame(locations)
    
    def generate_demand_data(self, num_days=365):
        """
        Generate daily demand data with various business metrics.
        
        Args:
            num_days (int): Number of days to generate data for
            
        Returns:
            pd.DataFrame: DataFrame containing daily demand data
        """
        data = []
        
        # Generate market trends
        trend = np.linspace(0, 1, num_days)  # Long-term growth trend
        market_noise = np.random.normal(0, 0.1, num_days)  # Market volatility
        competitor_impact = np.random.choice([-0.2, 0, 0.2], num_days, p=[0.2, 0.6, 0.2])  # Competitor actions
        
        # Generate supply chain disruptions
        disruption_prob = 0.05  # 5% chance of disruption
        disruption_impact = np.random.choice([1.0, 0.7, 0.5], num_days, p=[0.95, 0.03, 0.02])  # Supply chain disruptions
        
        for day in range(num_days):
            current_date = self.start_date + timedelta(days=day)
            
            # Calculate complex seasonality
            annual_season = 1 + 0.3 * np.sin(2 * np.pi * day / 365)  # Annual pattern
            weekly_season = 1 + 0.2 * np.sin(2 * np.pi * day / 7)    # Weekly pattern
            monthly_season = 1 + 0.15 * np.sin(2 * np.pi * day / 30) # Monthly pattern
            
            # Special events
            is_holiday = 1 if current_date.weekday() >= 5 or current_date.day == 1 else 0
            is_promotion = np.random.choice([0, 1], p=[0.8, 0.2])
            
            # Weather impact (simplified)
            weather_impact = np.random.normal(1, 0.1)
            
            for product in self.products.itertuples():
                for location in self.locations.itertuples():
                    # Base demand with product lifecycle
                    product_age = day / num_days
                    lifecycle_factor = 1.0
                    
                    # Calculate base demand
                    base_demand = product.base_demand * (
                        annual_season * 
                        weekly_season * 
                        monthly_season * 
                        disruption_impact[day] * 
                        weather_impact * 
                        lifecycle_factor
                    )
                    
                    # Add noise and trends
                    final_demand = base_demand * (
                        1 + trend[day] + 
                        market_noise[day] + 
                        competitor_impact[day]
                    )
                    
                    # Calculate inventory metrics with improved safety stock
                    target_inventory = final_demand * 1.2  # 20% safety stock
                    inventory_level = np.random.uniform(
                        max(target_inventory * 0.8, final_demand),  # Ensure minimum stock covers demand
                        target_inventory * 1.3
                    )
                    
                    # Calculate inventory turnover (annualized)
                    # Using quarterly rate for more stable numbers
                    quarterly_demand = final_demand * 90  # Quarterly demand
                    cost_of_goods_sold = quarterly_demand * product.base_price * 0.7  # Assuming 70% COGS
                    average_inventory_value = inventory_level * product.base_price
                    # Annualize the quarterly turnover rate (multiply by 4)
                    base_turnover = (cost_of_goods_sold * 4) / (average_inventory_value if average_inventory_value > 0 else 1)
                    # Apply industry-specific adjustments
                    if product.category == 'Food':
                        max_turnover = 15  # Higher for perishables
                    elif product.category == 'Electronics':
                        max_turnover = 8   # Medium for electronics
                    else:
                        max_turnover = 6   # Lower for general goods
                    
                    inventory_turnover = np.clip(base_turnover, 1, max_turnover)
                    
                    # Improve fulfillment rate calculation with safety stock consideration
                    stockout_risk = max(0, 1 - (inventory_level / final_demand))
                    fulfillment_rate = 1 - (stockout_risk * 0.5)  # Reduce impact of stockout risk
                    # Calculate fulfillment rate based on inventory
                    #fulfillment_rate = min(1.0, inventory_level / final_demand) if final_demand > 0 else 1.0
                    
                    # Promotion impact
                    price = product.base_price * (0.8 if is_promotion else 1.0)
                    
                    data.append({
                        'date': current_date,
                        'product_id': product.product_id,
                        'location_id': location.location_id,
                        'demand': final_demand,
                        'price': price,
                        'inventory_level': inventory_level,
                        'lead_time_days': np.random.randint(3, 14) * (1/disruption_impact[day]),  # Longer lead times during disruptions
                        'promotion_active': is_promotion,
                        'inventory_turnover': inventory_turnover,
                        'fulfillment_rate': fulfillment_rate
                    })
        
        return pd.DataFrame(data)
    
    def add_business_metrics(self, df):
        """
        Add derived business metrics to the dataset.
        
        Args:
            df (pd.DataFrame): Input DataFrame with base metrics
            
        Returns:
            pd.DataFrame: Enhanced DataFrame with additional business metrics
        """
        # Calculate base costs
        df['unit_cost'] = df['price'] * 0.6  # 60% of price is cost
        df['storage_cost_per_unit'] = df['price'] * 0.002  # 0.2% daily storage cost
        
        # Calculate inventory costs
        df['inventory_holding_cost'] = df['inventory_level'] * df['storage_cost_per_unit']
        df['stockout_cost'] = df['demand'] * df['price'] * (1 - df['fulfillment_rate']) * 1.5  # 1.5x price as penalty
        
        # Calculate revenue and costs
        df['potential_revenue'] = df['demand'] * df['price']
        df['actual_revenue'] = df['potential_revenue'] * df['fulfillment_rate']
        df['cogs'] = df['actual_revenue'] * 0.6  # 60% COGS
        df['total_cost'] = df['cogs'] + df['inventory_holding_cost'] + df['stockout_cost']
        
        # Calculate margins and profitability
        df['gross_profit'] = df['actual_revenue'] - df['cogs']
        df['operating_profit'] = df['actual_revenue'] - df['total_cost']
        df['profit_margin'] = df['operating_profit'] / df['actual_revenue']
        
        # Handle edge cases
        df['profit_margin'] = df['profit_margin'].fillna(0)  # Handle division by zero
        df['profit_margin'] = df['profit_margin'].clip(-1, 1)  # Cap margins between -100% and 100%
        
        return df

def main():
    """Example usage of the SupplyChainDataGenerator."""
    # Initialize generator
    generator = SupplyChainDataGenerator(
        start_date='2023-01-01',
        num_products=5,
        num_locations=3
    )
    
    # Generate one year of data
    data = generator.generate_demand_data(num_days=365)
    
    # Add business metrics
    data = generator.add_business_metrics(data)
    
    # Save to CSV
    data.to_csv('data/raw/supply_chain_data.csv', index=False)
    print(f"Generated {len(data)} records of supply chain data")
    print("\nSample of generated data:")
    print(data.head())
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(data.describe())

if __name__ == "__main__":
    main()