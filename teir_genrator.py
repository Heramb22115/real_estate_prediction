import pandas as pd
import json
import numpy as np

FILE_PATH = "india_housing_prices.csv"
TIER_MAPPING_FILENAME = 'city_tier_mapping.json'

def generate_tier_mapping(file_path):
    df = pd.read_csv(file_path)
    
    city_avg_price = df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False)
    
    q3 = city_avg_price.quantile(0.75)
    q1 = city_avg_price.quantile(0.25)
    
    def assign_tier(avg_price):
        if avg_price >= q3:
            return 'Tier_1'
        elif avg_price <= q1:
            return 'Tier_3'
        else:
            return 'Tier_2'
    
    tier_mapping_series = city_avg_price.apply(assign_tier)
    tier_mapping = tier_mapping_series.to_dict()
    
    with open(TIER_MAPPING_FILENAME, 'w') as f:
        json.dump(tier_mapping, f, indent=2)
    
    return TIER_MAPPING_FILENAME

if __name__ == "__main__":
    generated_file = generate_tier_mapping(FILE_PATH)
    print(f"City Tier mapping successfully generated and saved to {generated_file}")