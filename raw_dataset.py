import json
import pandas as pd

# Load your JSON list (assuming it's in a file)
with open("/Users/sami/Desktop/Projects/Dogo/polygon_news_sample.json", "r") as f:
    data = json.load(f)

rows = []

for article in data:
    article_id = article.get("id")
    base_desc = article.get("description", "")
    datetime = article.get("published_utc", "")[:10]  # YYYY-MM-DD
    
    insights = article.get("insights", [])
    
    # Expand one row per ticker
    for ins in insights:
        ticker = ins.get("ticker")
        sentiment = ins.get("sentiment")
        reasoning = ins.get("sentiment_reasoning", "")
        
        combined_desc = f"{base_desc} {reasoning}".strip()
        
        rows.append({
            "id": article_id,
            "datetime": datetime,
            "ticker": ticker,
            "description": combined_desc,
            "sentiment": sentiment
        })

# Create DataFrame
df = pd.DataFrame(rows)

print(df.head())

# Save DataFrame as CSV
df.to_csv("/Users/sami/Desktop/Projects/Dogo/raw_dataset.csv", index=False)
print(f"\nDataFrame saved to CSV file. Total rows: {len(df)}")