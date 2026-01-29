import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data_raw"
PROC_DIR = BASE_DIR / "data_processed"

# Create output folder
PROC_DIR.mkdir(exist_ok=True)

def prepare_marketing_data():
    print("Loading datasets...")
    
    # 1. Load campaigns (marketing spend)
    campaigns = pd.read_csv(RAW_DIR / "campaigns.csv")
    print("Campaigns shape:", campaigns.shape)
    print("Campaigns columns:", list(campaigns.columns))
    
    # 2. Load transactions (revenue)
    transactions = pd.read_csv(RAW_DIR / "transactions.csv")
    print("Transactions shape:", transactions.shape)
    
    # 3. Create date dimension (weekly from transaction timestamps)
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
    transactions['week_start'] = transactions['timestamp'].dt.to_period('W').dt.start_time.dt.date
    dim_date = pd.DataFrame({
        'date_key': transactions['week_start'].unique()
    }).sort_values('date_key').reset_index(drop=True)
    dim_date['year'] = pd.to_datetime(dim_date['date_key']).dt.year
    dim_date['week'] = pd.to_datetime(dim_date['date_key']).dt.isocalendar().week
    
    # 4. Create channel dimension from campaigns
    dim_channel = pd.DataFrame({
        'channel_name': campaigns['channel'].unique()
    }).reset_index(drop=True)
    dim_channel['channel_id'] = dim_channel.index + 1
    
    # 5. Prepare fact_marketing (spend per channel per week)
    # For simplicity: assume campaigns run for full weeks, assign budget to weeks
    # In real project you'd apportion by campaign dates
    fact_marketing_list = []
    for _, camp in campaigns.iterrows():
        camp_weeks = dim_date[dim_date['date_key'] >= pd.to_datetime(camp['start_date']).date()].head(4)  # ~1 month
        for week in camp_weeks['date_key']:
            fact_marketing_list.append({
                'date_key': week,
                'channel_id': dim_channel[dim_channel['channel_name'] == camp['channel']]['channel_id'].iloc[0],
                'budget': camp.get('budget', camp.get('expected_uplift', 0))  # Use budget or uplift as proxy
            })
    
    fact_marketing = pd.DataFrame(fact_marketing_list)
    fact_marketing = fact_marketing.groupby(['date_key', 'channel_id'])['budget'].sum().reset_index()
    fact_marketing.rename(columns={'budget': 'spend'}, inplace=True)
    
    # 6. Prepare fact_revenue (revenue per week, with campaign attribution)
    transactions_with_campaign = transactions.merge(
        campaigns[['campaign_id', 'channel']], on='campaign_id', how='left'
    )
    revenue_per_week = transactions_with_campaign.groupby('week_start')['gross_revenue'].sum().reset_index()
    revenue_per_week.rename(columns={'week_start': 'date_key', 'gross_revenue': 'revenue'}, inplace=True)
    
    # Save all tables
    dim_date.to_csv(PROC_DIR / "dim_date.csv", index=False)
    dim_channel.to_csv(PROC_DIR / "dim_channel.csv", index=False)
    fact_marketing.to_csv(PROC_DIR / "fact_marketing.csv", index=False)
    revenue_per_week.to_csv(PROC_DIR / "fact_revenue.csv", index=False)
    
    print("\n✅ Data prepared!")
    print("Saved files:")
    print("- dim_date.csv")
    print("- dim_channel.csv") 
    print("- fact_marketing.csv")
    print("- fact_revenue.csv")
    
    print("\nQuick check:")
    print("Channels:", dim_channel['channel_name'].tolist())
    print("Weeks:", len(dim_date))
    print("Marketing records:", len(fact_marketing))
    print("Revenue weeks:", len(revenue_per_week))

if __name__ == "__main__":
    prepare_marketing_data()
