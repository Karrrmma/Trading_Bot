# Trading_Bot
    A fully automated trading strategy built with the LumiBot framework and Alpaca API. This bot leverages sentiment analysis from financial news headlines to make intelligent buy/sell decisions on the SPY ETF using a bracket order strategy.

# Features
💬 Sentiment Analysis: Integrates a pre-trained model to estimate sentiment from financial news headlines.

🧠 ML-Driven Decision Making: Executes trades based on sentiment polarity and confidence.

💼 Risk Management: Trades only a specified percentage of available cash using bracket orders (take-profit + stop-loss).

🔄 Automated Scheduling: Makes decisions once per day, 15 minutes before market close.

🧪 Backtesting Support: Test historical performance with YahooDataBacktesting.

🛠 Tech Stack
lumibot - Automated trading framework

alpaca-trade-api - Paper/live trading with Alpaca

YahooDataBacktesting - Historical data backtesting

timedelta - Time calculation for data range

transformers-based model (via estimate_sentiment) for news analysis


# Install dependencies
pip install -r requirements.txt
# Add your Alpaca API credentials
Replace the placeholders in API_KEY, API_SECRET, and BASE_URL inside the 
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"
BASE_URL = "https://paper-api.alpaca.markets/v2"

Run the code
python ml_trader.py