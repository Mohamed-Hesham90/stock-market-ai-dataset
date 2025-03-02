from faker import Faker
import random
from datetime import datetime, timedelta
import json

# Initialize Faker
fake = Faker()

# Set random seed for reproducibility
random.seed(42)
Faker.seed(42)

def generate_fake_tweets(num_tweets=10000):
    """Generate fake tweets about Google with financial topics and emojis."""
    
    # Google-related financial keywords
    google_products = ['Google Search', 'Gmail', 'Google Cloud', 'Android', 'YouTube', 
                      'Google Maps', 'Google Workspace', 'Google AI', 'Google Ads', 
                      'Chrome', 'Pixel', 'Google Assistant']

    financial_terms = ['earnings', 'revenue', 'growth', 'profit', 'loss', 'stock', 
                       'market cap', 'valuation', 'acquisition', 'investment']

    # Define tweet patterns with emojis
    tweet_patterns = [
        "🚀 $GOOG is looking strong today! Huge gains incoming!",
        "🔥 $GOOG breaking out to new highs! Investors are excited!",
        "Google's {product} revenue surged by {percent}% 💰💡",
        "Massive earnings beat! Google reports {value} billion in revenue! 💵🚀",
        "Bullish on {product}, impressive performance! 📈🔥",
        "Google just acquired {company}, big move! 💼💡",
        "Strong demand for Google's {product}, stock is flying! 🚀💰",
        "Google stock up {percent}% after stellar earnings report! 📈💵",
        "Investors are loving Google's latest {product} innovation! 🔥💡",
        "Google under pressure after missing earnings! 📉",
        "⚠️ $GOOG breaking down, rough market reaction! 📉",
        "Google's {product} faces tough competition from {company}.",
        "Regulatory concerns hitting Google, potential lawsuits ahead! ⚖️📉",
        "Google layoffs reported in {product} division, stock down! 😞",
        "📢 Google announces updates for {product} at its latest event.",
        "📊 $GOOG trading sideways, waiting for earnings.",
        "💡 Google investing in {product}, interesting development!",
        "🤝 Google partnering with {company} on new tech initiative.",
        "📈 Google hiring aggressively in {product} sector.",
        "📢 Google conference reveals new {product} features."
    ]

    # Generate tweets
    tweets = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    for _ in range(num_tweets):
        # Choose a random tweet pattern
        pattern = random.choice(tweet_patterns)

        # Replace placeholders with fake data
        tweet = pattern.replace('{product}', random.choice(google_products))
        tweet = tweet.replace('{company}', fake.company())
        tweet = tweet.replace('{percent}', f"{random.randint(1, 15)}%")
        tweet = tweet.replace('{value}', f"{random.uniform(10, 50):.2f}")

        # Add hashtags (30% probability)
        if random.random() < 0.3:
            hashtags = random.sample(['#Google', '#GOOG', '#Stocks', '#Finance', '#TechStocks', '#Investing'], k=random.randint(1, 3))
            tweet += " " + " ".join(hashtags)

        # Add cashtags (40% probability)
        if random.random() < 0.4:
            tweet += " $GOOG"
        
        # Generate random timestamp within the past 90 days
        tweet_date = fake.date_time_between(start_date=start_date, end_date=end_date)

        # Simulate engagement metrics
        followers = random.randint(50, 500000)
        retweets = random.randint(0, 2000)
        likes = random.randint(0, 5000)

        # Append tweet data
        tweets.append({
            'username': fake.user_name(),
            'tweet_text': tweet,
            'timestamp': tweet_date.isoformat(),
            'followers': followers,
            'retweets': retweets,
            'likes': likes
        })

    return tweets

# Generate fake tweets
tweets_data = generate_fake_tweets(10000)

# Save to JSON file
with open('google_financial_tweets.json', 'w') as json_file:
    json.dump(tweets_data, json_file, indent=4)

print(f"Generated {len(tweets_data)} tweets and saved to 'google_financial_tweets.json'")
