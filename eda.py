import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

DATA_FILE = "india_housing_prices.csv"
TIER_MAPPING_FILE = "city_tier_mapping.json"

def load_and_clean_data(data_path, mapping_path):
    df = pd.read_csv(data_path)
    
    with open(mapping_path, 'r') as f:
        tier_mapping = json.load(f)
        
    df['City_Tier'] = df['City'].map(tier_mapping)
    df.drop(['ID', 'Locality', 'City'], axis=1, inplace=True)
    
    return df

def perform_eda(df):
    
    df['Log_Price'] = np.log1p(df['Price_in_Lakhs'])
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Price_in_Lakhs'], bins=50, kde=True)
    plt.title('1. Distribution of Property Prices (in Lakhs)')
    plt.savefig('eda_price_distribution.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='City_Tier', y='Price_in_Lakhs', data=df, order=['Tier_1', 'Tier_2', 'Tier_3'], palette='viridis')
    plt.title('2. Price Distribution by City Tier')
    plt.ylim(0, df['Price_in_Lakhs'].quantile(0.95))
    plt.savefig('eda_price_by_tier.png')
    plt.close()

    
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df.sample(n=10000), x='Size_in_SqFt', y='Log_Price', hue='City_Tier', fill=True, alpha=0.6)
    plt.title('3. Density Plot of Log(Price) vs. Size by City Tier')
    plt.ylabel('Log(1 + Price in Lakhs)')
    plt.savefig('eda_price_vs_size_density.png')
    plt.close()

    

if __name__ == "__main__":
    df = load_and_clean_data(DATA_FILE, TIER_MAPPING_FILE)
    perform_eda(df)
    print("EDA Complete. Check the generated PNG files for insights.")