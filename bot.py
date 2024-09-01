from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting #use for backtesting using historical data

from lumibot.strategies.strategy import Strategy#base trading class

from lumibot.traders import Trader
from datetime import datetime 
from alpaca_trade_api import REST 
from timedelta import Timedelta 
from sentiment import estimate_sentiment
# Replace finbert_utils with Hugging Face's transformers

API_KEY = "PKJKPQAUHK7O3JH8Q0AM" 
API_SECRET = "5vOaqSBUd7U6zn6y3eBXTRM37vh5Qf6UBiJ387er" 
BASE_URL = "https://paper-api.alpaca.markets/v2"

ALPACA_CREDS = {
    "API_KEY":API_KEY, 
    "API_SECRET": API_SECRET, 
    "PAPER": True
}

class MLTrader(Strategy): 
    def initialize(self, symbol:str="SPY", cash_at_risk:float=.5): 
        self.symbol = symbol
        self.sleeptime = "24H"
        self.minutes_before_closing = 15 
        
        self.last_trade = None 
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

    def position_sizing(self): 
        cash = self.get_cash() 
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price,0)
        return cash, last_price, quantity
    
    def before_market_opens(self):
        bars_list =  self.get_historical_prices_for_assets(["SPY"], 2, "day")
        for asset_bars in bars_list:
            self.log_message("Asset bars: {asset_bars.__dict__}")

    def get_dates(self): 
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    def get_sentiment(self): 
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, 
                                 start=three_days_prior, 
                                 end=today) 
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment 
    def after_market_closes(self):
        self.log_message(f"The total value of our portforlio is {self.portfolio_value}")
    def on_bot_crash(self, error):
        self.sell_all()

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing() 
        probability, sentiment = self.get_sentiment()

        if cash > last_price: 
            if sentiment == "positive" and probability > .999: 
                if self.last_trade == "sell": 
                    self.sell_all() 
                order = self.create_order(
                    self.symbol, 
                    quantity, 
                    "buy", 
                    type="bracket", 
                    take_profit_price=last_price*1.20, 
                    stop_loss_price=last_price*.80
                )
                self.submit_order(order) 
                self.last_trade = "buy"
            elif sentiment == "negative" and probability > .999: 
                if self.last_trade == "buy": 
                    self.sell_all() 
                order = self.create_order(
                    self.symbol, 
                    quantity, 
                    "sell", 
                    type="bracket", 
                    take_profit_price=last_price*.8, 
                    stop_loss_price=last_price*1.05
                )
                self.submit_order(order) 
                self.last_trade = "sell"

start_date = datetime(2023,1,1)
end_date = datetime(2023,12,31) 
#connects to the api
broker = Alpaca(ALPACA_CREDS) 
strategy = MLTrader(name='TradeStrategy', broker=broker, 
                    parameters={"symbol":"SPY", 
                                "cash_at_risk":.4})
strategy.backtest(
    YahooDataBacktesting, 
    start_date, 
    end_date, 
    parameters={"symbol":"SPY", "cash_at_risk":.5}
)
trader = Trader()
trader.add_strategy(strategy)
trader.run_all()