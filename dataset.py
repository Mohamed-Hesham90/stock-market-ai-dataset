import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import time
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
import tweepy

class FinancialSentimentCollector:
    def __init__(self, output_dir="financial_sentiment_data", 
                 twitter_api_key=None, twitter_api_secret=None, 
                 twitter_access_token=None, twitter_access_secret=None,
                 newsapi_key=None):
        """
        Initialize the Financial Sentiment Data Collector.
        
        Args:
            output_dir (str): Directory to store collected data
            twitter_api_key (str): Twitter API key (optional)
            twitter_api_secret (str): Twitter API secret (optional)
            twitter_access_token (str): Twitter access token (optional)
            twitter_access_secret (str): Twitter access token secret (optional)
            newsapi_key (str): NewsAPI key (optional)
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create subdirectories
        self.price_dir = os.path.join(output_dir, "price_data")
        self.news_dir = os.path.join(output_dir, "news_sentiment")
        self.social_dir = os.path.join(output_dir, "social_sentiment")
        self.combined_dir = os.path.join(output_dir, "combined_data")
        
        for directory in [self.price_dir, self.news_dir, self.social_dir, self.combined_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Set up API credentials
        self.twitter_credentials = None
        if all([twitter_api_key, twitter_api_secret, twitter_access_token, twitter_access_secret]):
            self.twitter_credentials = {
                "api_key": twitter_api_key,
                "api_secret": twitter_api_secret,
                "access_token": twitter_access_token,
                "access_secret": twitter_access_secret
            }
        
        self.newsapi_key = newsapi_key
        
        # Set up NLTK for sentiment analysis
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
            nltk.download('punkt')
        
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Add finance-specific terms to VADER lexicon to improve sentiment scoring
        self.update_vader_lexicon()
    
    def update_vader_lexicon(self):
        """Update VADER lexicon with finance-specific terms"""
        finance_lexicon = {
            # Positive financial terms
            'bullish': 2.5, 'outperform': 2.0, 'buy': 2.0, 'upgrade': 2.0, 
            'beat': 1.5, 'exceeded': 1.5, 'profit': 1.5, 'growth': 1.5,
            'upside': 1.5, 'dividend': 1.0, 'uptrend': 1.5, 'rally': 1.5,
            
            # Negative financial terms
            'bearish': -2.5, 'underperform': -2.0, 'sell': -2.0, 'downgrade': -2.0,
            'miss': -1.5, 'below': -1.0, 'loss': -2.0, 'debt': -1.0,
            'downside': -1.5, 'crash': -3.0, 'downtrend': -1.5, 'bankruptcy': -3.0,
            'recession': -2.5, 'inflation': -1.0, 'volatility': -0.5
        }
        
        # Update the lexicon
        self.sentiment_analyzer.lexicon.update(finance_lexicon)
    
    def _get_stock_list(self, list_type="major"):
        """Get a list of stock tickers based on the specified type."""
        if list_type == "major":
            # Major stocks that are often discussed in financial news and social media
            return ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "BAC", "WMT"]
        elif list_type == "tech":
            # Technology stocks with high social media presence
            return ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "TSLA", "NFLX", "CRM", "ADBE", "INTC", "AMD", "PYPL", "UBER", "ABNB"]
        elif list_type == "finance":
            # Financial stocks
            return ["JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "AXP", "V", "MA", "COF", "SCHW"]
        elif list_type == "volatile":
            # Stocks with higher volatility that might show stronger sentiment effects
            return ["TSLA", "GME", "AMC", "COIN", "RIVN", "DKNG", "PLTR", "NIO", "SNAP", "RBLX", "HOOD", "SPCE"]
        elif list_type == "meme":
            # Meme stocks with high social media activity
            return ["GME", "AMC", "BB", "EXPR", "KOSS", "NOK", "BBBY", "WISH", "CLOV", "MVIS", "TLRY", "SNDL"]
        else:
            # Default to major stocks
            return ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "BAC", "WMT"]
    
    def _get_crypto_list(self, list_type="major"):
        """Get a list of cryptocurrency tickers based on the specified type."""
        if list_type == "major":
            # Major cryptocurrencies
            return ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "DOT", "AVAX", "MATIC"]
        elif list_type == "meme":
            # Meme cryptocurrencies with high social sentiment volatility
            return ["DOGE", "SHIB", "PEPE", "FLOKI", "BONK", "ELON", "SAMO", "WIF", "MONA", "BABYDOGE"]
        else:
            # Default to major cryptocurrencies
            return ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "DOT", "AVAX", "MATIC"]
    
    def collect_price_data(self, ticker, asset_type, period="30d", interval="1h"):
        """Collect price data with finer granularity for real-time sentiment analysis."""
        try:
            ticker_obj = yf.Ticker(ticker if asset_type != "crypto" else f"{ticker}-USD")
            hist = ticker_obj.history(period=period, interval=interval)
            
            if hist.empty:
                return {"error": f"No historical data available for {ticker}"}
            
            # Convert to structured format
            price_data = []
            for date, row in hist.iterrows():
                data_point = {
                    "timestamp": date.strftime("%Y-%m-%d %H:%M:%S"),
                    "open": float(row["Open"]) if not np.isnan(row["Open"]) else None,
                    "high": float(row["High"]) if not np.isnan(row["High"]) else None,
                    "low": float(row["Low"]) if not np.isnan(row["Low"]) else None,
                    "close": float(row["Close"]) if not np.isnan(row["Close"]) else None,
                    "volume": int(row["Volume"]) if not np.isnan(row["Volume"]) else None
                }
                price_data.append(data_point)
            
            # Calculate short-term technical indicators relevant for sentiment analysis
            if len(price_data) > 10:
                closes = hist["Close"].values
                volumes = hist["Volume"].values
                
                # Calculate rolling volatility (5-period)
                for i in range(5, len(price_data)):
                    window = closes[i-5:i]
                    volatility = float(np.std(window) / np.mean(window) * 100)  # Percentage volatility
                    price_data[i]["volatility_5period"] = round(volatility, 2)
                
                # Calculate momentum (change over last 5 periods)
                for i in range(5, len(price_data)):
                    momentum = float((closes[i] - closes[i-5]) / closes[i-5] * 100)
                    price_data[i]["momentum_5period"] = round(momentum, 2)
                
                # Calculate volume surge (ratio to 5-period average)
                for i in range(5, len(price_data)):
                    avg_volume = np.mean(volumes[i-5:i])
                    if avg_volume > 0:
                        vol_ratio = float(volumes[i] / avg_volume)
                        price_data[i]["volume_ratio_5period"] = round(vol_ratio, 2)
            
            return {
                "ticker": ticker,
                "asset_type": asset_type,
                "interval": interval,
                "price_data": price_data,
                "metadata": {
                    "period": period,
                    "data_points": len(price_data),
                    "start_time": price_data[0]["timestamp"] if price_data else None,
                    "end_time": price_data[-1]["timestamp"] if price_data else None,
                    "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}
    
    def collect_news_sentiment(self, ticker, asset_type, days_back=7):
        """Collect news sentiment for a ticker over the specified period."""
        if not self.newsapi_key:
            # Try to get financial news from alternative sources if NewsAPI key not available
            return self.collect_alternative_news(ticker, asset_type, days_back)
        
        try:
            # Prepare for API call
            if asset_type == "crypto":
                if ticker == "BTC":
                    query = "Bitcoin OR BTC"
                elif ticker == "ETH":
                    query = "Ethereum OR ETH"
                else:
                    query = f"{ticker} cryptocurrency"
            else:
                company_name = yf.Ticker(ticker).info.get("shortName", ticker)
                query = f"({ticker} OR {company_name}) stock"
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Make API request to NewsAPI
            url = f"https://newsapi.org/v2/everything?q={query}&from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&language=en&sortBy=publishedAt&apiKey={self.newsapi_key}"
            response = requests.get(url)
            
            if response.status_code != 200:
                return self.collect_alternative_news(ticker, asset_type, days_back)
            
            news_data = response.json()
            if news_data.get("status") != "ok" or news_data.get("totalResults", 0) == 0:
                return self.collect_alternative_news(ticker, asset_type, days_back)
            
            # Process articles
            articles = []
            for article in news_data.get("articles", []):
                # Combine title and description for sentiment analysis
                text = f"{article.get('title', '')} {article.get('description', '')}"
                
                # Skip if no meaningful text
                if len(text.strip()) < 20:
                    continue
                
                # Get sentiment scores
                sentiment = self.sentiment_analyzer.polarity_scores(text)
                
                # Create article entry
                article_data = {
                    "source": article.get("source", {}).get("name"),
                    "title": article.get("title"),
                    "published_at": article.get("publishedAt"),
                    "url": article.get("url"),
                    "sentiment": {
                        "compound": sentiment["compound"],
                        "positive": sentiment["pos"],
                        "negative": sentiment["neg"],
                        "neutral": sentiment["neu"],
                        "label": "positive" if sentiment["compound"] >= 0.05 else 
                                 "negative" if sentiment["compound"] <= -0.05 else "neutral"
                    }
                }
                articles.append(article_data)
            
            # Group articles by day for time series analysis
            daily_sentiment = {}
            for article in articles:
                try:
                    date = article["published_at"].split("T")[0]
                    if date not in daily_sentiment:
                        daily_sentiment[date] = {
                            "articles": 0,
                            "sentiment_sum": 0,
                            "positive_count": 0,
                            "negative_count": 0,
                            "neutral_count": 0
                        }
                    
                    daily_sentiment[date]["articles"] += 1
                    daily_sentiment[date]["sentiment_sum"] += article["sentiment"]["compound"]
                    
                    if article["sentiment"]["label"] == "positive":
                        daily_sentiment[date]["positive_count"] += 1
                    elif article["sentiment"]["label"] == "negative":
                        daily_sentiment[date]["negative_count"] += 1
                    else:
                        daily_sentiment[date]["neutral_count"] += 1
                except Exception:
                    continue
            
            # Calculate daily average sentiment
            daily_averages = []
            for date, data in daily_sentiment.items():
                if data["articles"] > 0:
                    avg_sentiment = data["sentiment_sum"] / data["articles"]
                    daily_averages.append({
                        "date": date,
                        "articles_count": data["articles"],
                        "avg_sentiment": round(avg_sentiment, 3),
                        "positive_ratio": round(data["positive_count"] / data["articles"], 3),
                        "negative_ratio": round(data["negative_count"] / data["articles"], 3),
                        "neutral_ratio": round(data["neutral_count"] / data["articles"], 3)
                    })
            
            return {
                "ticker": ticker,
                "asset_type": asset_type,
                "news_sentiment": {
                    "daily_averages": sorted(daily_averages, key=lambda x: x["date"]),
                    "articles": articles
                },
                "metadata": {
                    "total_articles": len(articles),
                    "period_days": days_back,
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}
    
    def collect_alternative_news(self, ticker, asset_type, days_back=7):
        """Fallback method to collect financial news from public sources."""
        try:
            articles = []
            
            # Define search URL based on asset type
            if asset_type == "crypto":
                if ticker == "BTC":
                    query = "Bitcoin+price"
                elif ticker == "ETH":
                    query = "Ethereum+price"
                else:
                    query = f"{ticker}+crypto+price"
                
                # Try to scrape from CoinDesk or similar sites
                urls = [
                    f"https://www.coindesk.com/search?q={ticker.lower()}",
                    f"https://cointelegraph.com/tags/{ticker.lower()}"
                ]
            else:
                # Try to scrape from MarketWatch or similar sites
                urls = [
                    f"https://www.marketwatch.com/search?q={ticker}&ts=0&tab=Articles"
                ]
            
            for url in urls:
                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                    }
                    response = requests.get(url, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Extract article elements (simplified - real implementation would need site-specific parsing)
                        article_elements = soup.find_all('article') or soup.find_all('div', class_=lambda x: x and ('article' in x.lower() or 'story' in x.lower()))
                        
                        for element in article_elements[:20]:  # Limit to first 20 articles
                            title_element = element.find('h2') or element.find('h3')
                            if not title_element:
                                continue
                                
                            title = title_element.get_text().strip()
                            
                            # Calculate sentiment
                            sentiment = self.sentiment_analyzer.polarity_scores(title)
                            
                            # Try to find publication date
                            date_element = element.find('time') or element.find('span', class_=lambda x: x and ('date' in x.lower() or 'time' in x.lower()))
                            pub_date = date_element.get_text().strip() if date_element else "N/A"
                            
                            # Create article entry
                            article_data = {
                                "source": url.split('/')[2],
                                "title": title,
                                "published_at": pub_date,
                                "url": "N/A",  # Would extract actual URL in real implementation
                                "sentiment": {
                                    "compound": sentiment["compound"],
                                    "positive": sentiment["pos"],
                                    "negative": sentiment["neg"],
                                    "neutral": sentiment["neu"],
                                    "label": "positive" if sentiment["compound"] >= 0.05 else 
                                             "negative" if sentiment["compound"] <= -0.05 else "neutral"
                                }
                            }
                            articles.append(article_data)
                            
                except Exception:
                    continue
            
            # If we couldn't get articles, return a status
            if not articles:
                return {
                    "ticker": ticker,
                    "status": "No articles found through alternative sources",
                    "metadata": {
                        "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                }
                
            # Return collected articles with sentiment
            return {
                "ticker": ticker,
                "asset_type": asset_type,
                "news_sentiment": {
                    "articles": articles
                },
                "metadata": {
                    "total_articles": len(articles),
                    "note": "Data collected through alternative sources - limited reliability",
                    "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
        except Exception as e:
            return {"ticker": ticker, "error": f"Alternative news collection failed: {str(e)}"}
    
    def collect_social_sentiment(self, ticker, asset_type, days_back=3):
        """Collect social media sentiment for a ticker."""
        if not self.twitter_credentials:
            return {"ticker": ticker, "status": "Twitter API credentials not provided"}
        
        try:
            # Set up Twitter API client
            auth = tweepy.OAuthHandler(
                self.twitter_credentials["api_key"], 
                self.twitter_credentials["api_secret"]
            )
            auth.set_access_token(
                self.twitter_credentials["access_token"], 
                self.twitter_credentials["access_secret"]
            )
            api = tweepy.API(auth)
            
            # Define search query
            if asset_type == "crypto":
                if ticker == "BTC":
                    query = "$BTC OR #Bitcoin OR Bitcoin"
                elif ticker == "ETH":
                    query = "$ETH OR #Ethereum OR Ethereum"
                else:
                    query = f"${ticker} OR #{ticker}"
            else:
                query = f"${ticker} OR #{ticker} stock"
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Collect tweets
            tweets = []
            for tweet in tweepy.Cursor(api.search_tweets, q=query, lang="en", 
                                       tweet_mode="extended", count=100).items(500):
                created_at = tweet.created_at
                
                if created_at < start_date:
                    continue
                
                # Clean text
                text = tweet.full_text
                text = re.sub(r'http\S+', '', text)  # Remove URLs
                text = re.sub(r'@\w+', '', text)     # Remove mentions
                text = re.sub(r'RT\s+', '', text)    # Remove RT
                text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
                text = text.strip()
                
                if len(text) < 10:  # Skip very short tweets
                    continue
                
                # Calculate sentiment
                sentiment = self.sentiment_analyzer.polarity_scores(text)
                
                # Create tweet entry
                tweet_data = {
                    "id": tweet.id_str,
                    "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "text": text,
                    "user_followers": tweet.user.followers_count,
                    "retweet_count": tweet.retweet_count,
                    "favorite_count": tweet.favorite_count,
                    "sentiment": {
                        "compound": sentiment["compound"],
                        "positive": sentiment["pos"],
                        "negative": sentiment["neg"],
                        "neutral": sentiment["neu"],
                        "label": "positive" if sentiment["compound"] >= 0.05 else 
                                 "negative" if sentiment["compound"] <= -0.05 else "neutral"
                    }
                }
                tweets.append(tweet_data)
            
            # Group tweets by day and hour for time series analysis
            hourly_sentiment = {}
            for tweet in tweets:
                try:
                    # Get date and hour
                    created_dt = datetime.strptime(tweet["created_at"], "%Y-%m-%d %H:%M:%S")
                    date_hour = created_dt.strftime("%Y-%m-%d %H")
                    
                    if date_hour not in hourly_sentiment:
                        hourly_sentiment[date_hour] = {
                            "tweets": 0,
                            "sentiment_sum": 0,
                            "sentiment_weighted_sum": 0,
                            "total_weight": 0,
                            "positive_count": 0,
                            "negative_count": 0,
                            "neutral_count": 0
                        }
                    
                    # Calculate tweet importance weight based on engagement
                    weight = 1 + 0.1 * (tweet["retweet_count"] + tweet["favorite_count"]) + 0.001 * tweet["user_followers"]
                    
                    hourly_sentiment[date_hour]["tweets"] += 1
                    hourly_sentiment[date_hour]["sentiment_sum"] += tweet["sentiment"]["compound"]
                    hourly_sentiment[date_hour]["sentiment_weighted_sum"] += tweet["sentiment"]["compound"] * weight
                    hourly_sentiment[date_hour]["total_weight"] += weight
                    
                    if tweet["sentiment"]["label"] == "positive":
                        hourly_sentiment[date_hour]["positive_count"] += 1
                    elif tweet["sentiment"]["label"] == "negative":
                        hourly_sentiment[date_hour]["negative_count"] += 1
                    else:
                        hourly_sentiment[date_hour]["neutral_count"] += 1
                        
                except Exception:
                    continue
            
            # Calculate hourly average sentiment
            hourly_averages = []
            for date_hour, data in hourly_sentiment.items():
                if data["tweets"] > 0:
                    avg_sentiment = data["sentiment_sum"] / data["tweets"]
                    weighted_avg = data["sentiment_weighted_sum"] / data["total_weight"] if data["total_weight"] > 0 else avg_sentiment
                    
                    hourly_averages.append({
                        "date_hour": date_hour,
                        "tweets_count": data["tweets"],
                        "avg_sentiment": round(avg_sentiment, 3),
                        "weighted_sentiment": round(weighted_avg, 3),
                        "positive_ratio": round(data["positive_count"] / data["tweets"], 3),
                        "negative_ratio": round(data["negative_count"] / data["tweets"], 3),
                        "neutral_ratio": round(data["neutral_count"] / data["tweets"], 3)
                    })
            
            return {
                "ticker": ticker,
                "asset_type": asset_type,
                "social_sentiment": {
                    "hourly_averages": sorted(hourly_averages, key=lambda x: x["date_hour"]),
                    "tweets": tweets[:100]  # Limit to first 100 tweets in output
                },
                "metadata": {
                    "total_tweets": len(tweets),
                    "period_days": days_back,
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
        except Exception as e:
            return {"ticker": ticker, "error": f"Social sentiment collection failed: {str(e)}"}
    
    def collect_sentiment_data_batch(self, tickers, asset_type, collect_price=True, 
                                   collect_news=True, collect_social=True):
        """Collect all sentiment data for a batch of tickers."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Collect price data
            if collect_price:
                print(f"Collecting price data for {len(tickers)} {asset_type}s...")
                future_to_ticker = {
                    executor.submit(self.collect_price_data, ticker, asset_type): ticker 
                    for ticker in tickers
                }
                
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        data = future.result()
                        if "error" not in data:
                            # Save individual ticker data
                            file_path = os.path.join(self.price_dir, f"{ticker}_price.json")
                            with open(file_path, 'w') as f:
                                json.dump(data, f, indent=2)
                            
                            if ticker not in results:
                                results[ticker] = {}
                            results[ticker]["price"] = data
                            print(f"✓ Saved price data for {ticker}")
                        else:
                            print(f"✗ Error with price data for {ticker}: {data['error']}")
                            if ticker not in results:
                                results[ticker] = {}
                            results[ticker]["price_error"] = data["error"]
                    except Exception as e:
                        print(f"✗ Error processing price data for {ticker}: {str(e)}")
                        if ticker not in results:
                            results[ticker] = {}
                        results[ticker]["price_error"] = str(e)
                    
                    # Add a small delay
                    time.sleep(0.5)
            
            # Collect news sentiment
            if collect_news:
                print(f"Collecting news sentiment for {len(tickers)} {asset_type}s...")
                future_to_ticker = {
                    executor.submit(self.collect_news_sentiment, ticker, asset_type): ticker 
                    for ticker in tickers
                }
                
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        data = future.result()
                        if "error" not in data:
                            # Save individual ticker data
                            file_path = os.path.join(self.news_dir, f"{ticker}_news.json")
                            with open(file_path, 'w') as f:
                                json.dump(data, f, indent=2)
                            
                            if ticker not in results:
                                results[ticker] = {}
                            results[ticker]["news"] = data
                            print(f"✓ Saved news sentiment for {ticker}")
                        else:
                            print(f"✗ Error with news sentiment for {ticker}: {data['error']}")
                            if ticker not in results:
                                results[ticker] = {}
                            results[ticker]["news_error"] = data["error"]
                    except Exception as e:
                        print(f"✗ Error processing news sentiment for {ticker}: {str(e)}")
                        if ticker not in results:
                            results[ticker] = {}
                        results[ticker]["news_error"] = str(e)
                    
                    # Add a small delay
                    time.sleep(1)
            
            # Collect social sentiment
            if collect_social and self.twitter_credentials:
                print(f"Collecting social sentiment for {len(tickers)} {asset_type}s...")
                future_to_ticker = {
                    executor.submit(self.collect_social_sentiment, ticker, asset_type): ticker 
                    for ticker in tickers
                }
                
                for future in as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        data = future.result()
                        if "error" not in data:
                            # Save individual ticker data
                            file_path = os.path.join(self.social_dir, f"{ticker}_social.json")
                            with open(file_path, 'w') as f:
                                json.dump(data, f, indent=2)
                            
                            if ticker not in results:
                                results[ticker] = {}
                            results[ticker]["social"] = data
                            print(f"✓ Saved social sentiment for {ticker}")
                        else:
                            print(f"✗ Error with social sentiment for {ticker}: {data['error']}")
                            if ticker not in results:
                                results[ticker] = {}
                            results[ticker]["social_error"] = data["error"]
                    except Exception as e:
                        print(f"✗ Error processing social sentiment for {ticker}: {str(e)}")
                        if ticker not in results:
                            results[ticker] = {}
                        results[ticker]["social_error"] = str(e)
                    
                    # Add a small delay to avoid rate limits
                    time.sleep(2)