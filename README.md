# stock-market-ai-dataset
AI training dataset for stock market sentiment analysis using fake tweets and real market data.
📌 Project Overview
The Stock Market AI Dataset project is designed to generate and collect data for training AI models in stock market sentiment analysis. It consists of two key components:

1️⃣ Fake Tweet Generator (fakedata.py)

Creates synthetic stock market tweets for AI model training.
Stores generated tweets in a JSON file.
2️⃣ Stock Market Data Collector (dataset.py)

Gathers real-time stock market data from various sources.
Organizes and stores information about all farms (companies) in the stock market.
This dataset can be used for AI-powered sentiment analysis, market trend prediction, and algorithmic trading research. 🚀
📌 Features – Main Functionalities
✅ Fake Tweet Generator (fakedata.py)

Generates synthetic stock market tweets for AI training.
Simulates positive, negative, and neutral sentiments.
Stores tweets in a JSON file for dataset preparation.
✅ Stock Market Data Collector (dataset.py)

Collects real-time stock market data.
Gathers information about all farms (companies) in the stock market.
Stores structured data for further analysis and AI model training.
✅ AI Training Dataset Preparation

Creates a high-quality dataset for stock market sentiment analysis.
Helps in machine learning model training for financial predictions.
📌 Installation – How to Set It Up
Follow these steps to set up and run the project:
1️⃣ Clone the Repository
bash:

git clone https://github.com/Mohamed-Hesham90/stock-market-ai-dataset.git
cd stock-market-ai-dataset

2️⃣ Create a Virtual Environment (Optional but Recommended)

python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows

📌 Dependencies – Required Libraries
Here are the dependencies needed for your project:

bash

pip install faker pandas numpy requests yfinance nltk beautifulsoup4 tweepy

📜 Breakdown of Dependencies:
faker – Generates fake stock market tweets
pandas – Handles data processing and storage
numpy – Supports numerical operations
requests – Fetches data from APIs and web sources
yfinance – Retrieves real-time stock market data
nltk – Performs sentiment analysis on stock-related text
beautifulsoup4 – Parses HTML content from web scraping
tweepy – Connects to Twitter API for real tweet collection


4️⃣ Run the Scripts

Generate Fake Tweets:
bash

python fakedata.py


Collect Stock Market Data:
bash

python dataset.py

