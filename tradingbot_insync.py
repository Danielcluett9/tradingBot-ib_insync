import threading
from ib_insync import *
import pandas as pd
import numpy as np
import yfinance as yf
import time
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.momentum import StochRSIIndicator
from dataclasses import dataclass, field
from typing import List, ClassVar

@dataclass
class Trade:
    events: ClassVar = (
        'statusEvent', 'modifyEvent', 'fillEvent',
        'commissionReportEvent', 'filledEvent',
        'cancelEvent', 'cancelledEvent')

    contract: Contract = field(default_factory=Contract)
    order: Order = field(default_factory=Order)
    orderStatus: 'OrderStatus' = field(default_factory=OrderStatus)
    fills: List[Fill] = field(default_factory=list)
    log: List[TradeLogEntry] = field(default_factory=list)
    advancedError: str = ''

    def __post_init__(self):
        self.statusEvent = Event('statusEvent')
        self.modifyEvent = Event('modifyEvent')
        self.fillEvent = Event('fillEvent')
        self.commissionReportEvent = Event('commissionReportEvent')
        self.filledEvent = Event('filledEvent')
        self.cancelEvent = Event('cancelEvent')
        self.cancelledEvent = Event('cancelledEvent')

    def isActive(self):
        return self.orderStatus.status in OrderStatus.ActiveStates

    def isDone(self):
        return self.orderStatus.status in OrderStatus.DoneStates

    def filled(self):
        fills = self.fills
        if self.contract.secType == 'BAG':
            fills = [f for f in fills if f.contract.secType == 'BAG']
        return sum(f.execution.shares for f in fills)

    def remaining(self):
        return self.order.totalQuantity - self.filled()


class TradingBot:
    def __init__(self, symbols, client_id=1, port=7497, bar_size='5 mins'):
        self.symbols = symbols
        self.bar_size = bar_size
        self.ib = IB()
        util.startLoop()
        self.ib.connect(port=port, clientId=client_id)
        self.contracts = [Stock(symbol=symbol, exchange='SMART', currency='USD') for symbol in symbols]
        self.historical_data = {symbol: None for symbol in symbols}
        self.live_data = {symbol: None for symbol in symbols}
        self.combined_data = {symbol: pd.DataFrame() for symbol in symbols}
        self.positions = {symbol: 0 for symbol in symbols}
        self.trade_flags = {symbol: None for symbol in symbols}
        self.pending_orders = {symbol: None for symbol in symbols}
        self.active_trades = {symbol: None for symbol in symbols}  # Track active trades
        self.five_second_bars = {symbol: pd.DataFrame() for symbol in symbols}  # Store 5-second bars

    def qualify_contracts(self):
        for contract in self.contracts:
            self.ib.qualifyContracts(contract)

    def fetch_historical_data(self):
        for symbol in self.symbols:
            try:
                df = yf.download(symbol, period='1mo', interval='5m')
                df.index.name = 'date'
                df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
                self.historical_data[symbol] = df
                print(f"Fetched historical data for {symbol}")
            except Exception as e:
                print(f"Error fetching historical data for {symbol}: {e}")
        return self.historical_data

    def fetch_live_data(self):
        bar_size_in_seconds = {
            '5 secs': 5,
            '1 min': 60,
            '2 mins': 120,
            '3 mins': 180,
            '5 mins': 300,
            '15 mins': 900,
            '60 mins': 3600
        }
        for contract in self.contracts:
            symbol = contract.symbol
            try:
                self.live_data[symbol] = self.ib.reqRealTimeBars(
                    contract=contract,
                    barSize=bar_size_in_seconds['5 secs'],  # Always request 5-second bars
                    whatToShow="TRADES",
                    useRTH=False
                )
                self.live_data[symbol].updateEvent += self.create_on_bar_update(symbol)
                print(f"Fetching live data for {symbol}")
            except Exception as e:
                print(f"Error fetching live data for {symbol}: {e}")
        return {symbol: util.df(bars) for symbol, bars in self.live_data.items() if bars}

    def create_on_bar_update(self, symbol):
        def on_bar_update(bars, hasNewBar):
            if hasNewBar:
                new_bar = {
                    'time': bars[-1].time,
                    'open': bars[-1].open_,
                    'high': bars[-1].high,
                    'low': bars[-1].low,
                    'close': bars[-1].close,
                    'volume': bars[-1].volume,
                    'wap': bars[-1].wap
                }
                new_bar_df = pd.DataFrame([new_bar]).set_index('time')
                self.five_second_bars[symbol] = pd.concat([self.five_second_bars[symbol], new_bar_df])
                self.aggregate_bars(symbol)
                self.run_strategy(symbol)
                print(f"New bar added for {symbol}: {new_bar}")
        return on_bar_update

    def combine_data(self):
        for symbol in self.symbols:
            if self.historical_data[symbol] is not None:
                historical_df = self.historical_data[symbol]
                self.combined_data[symbol] = historical_df
                print(f"Initial historical data for {symbol}:")
                print(self.combined_data[symbol].head())
        return self.combined_data

    def aggregate_bars(self, symbol):
        # Resample the 5-second bars into 5-minute bars
        resampled_df = self.five_second_bars[symbol].resample('5T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'wap': 'mean'
        }).dropna()
        
        # Update the combined data
        self.combined_data[symbol] = pd.concat([self.combined_data[symbol], resampled_df])
        self.combined_data[symbol] = self.combined_data[symbol].sort_index()
        print(f"Updated combined data for {symbol}:")
        print(self.combined_data[symbol].tail())

    def update_df(self, symbol):
        if self.live_data[symbol] is None:
            print(f"No live data available for {symbol}")
            return

    def run_strategy(self, symbol):
        df = self.combined_data[symbol]
        if df.empty or len(df) < 2:
            return
        
        ma_window = 5
        long_ma_window = 20
        
        stoch_rsi = StochRSIIndicator(close=df['close'], window=14, smooth1=3, smooth2=3, fillna=False)
        rsi = RSIIndicator(close=df['close'], window=14)
        bbands = BollingerBands(close=df['close'], window=20, window_dev=1)
        bbands2 = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['mavg'] = df['close'].rolling(window=ma_window, min_periods=1).mean()
        df['long_mavg'] = df['close'].rolling(window=long_ma_window, min_periods=1).mean()
        
        df['Mband'] = bbands.bollinger_mavg()
        df['Hband'] = bbands.bollinger_hband()
        df['Lband'] = bbands.bollinger_lband()
        df['Mband2'] = bbands2.bollinger_mavg()
        df['Hband2'] = bbands2.bollinger_hband()
        df['Lband2'] = bbands2.bollinger_lband()
        df['rsi'] = rsi.rsi()
        df['stochrsi'] = stoch_rsi.stochrsi()
        df['stoch%d'] = stoch_rsi.stochrsi_d()
        df['stoch%k'] = stoch_rsi.stochrsi_k()

        latest = df.iloc[-1]
        previous = df.iloc[-2]

        if self.is_trade_active[symbol] is None and self.trade_flags[symbol] is None:
            if previous['close'] < df['Hband'] and latest['close'] >= df['Hband']:
                self.place_order(symbol, 'BUY', stop_loss=df['Hband'])
                self.positions[symbol] = 1
                self.trade_flags[symbol] = 'BANDS'
            elif latest['close'] > latest['mavg'] and latest['mavg'] > previous['mavg']:
                self.place_order(symbol, 'BUY', stop_loss=latest['mavg'])
                self.positions[symbol] = 1
                self.trade_flags[symbol] = 'TREND'
            elif latest['rsi'] <= 30 and latest['close'] > df['Mband']:
                self.place_order(symbol, 'BUY', stop_loss=df['Mband'])
                self.positions[symbol] = 1
                self.trade_flags[symbol] = 'RSI'
            elif latest['stochrsi'] <= .20 and latest['mavg'] > latest['long_mavg']:
                self.place_order(symbol, 'BUY', stop_loss=latest['long_mavg'])
                self.positions[symbol] = smooth1
                self.trade_flags[symbol] = 'STOCHRSI'
               
    def is_trade_active(self, symbol):
        trade = self.active_trades.get(symbol)
        return trade is not None and not trade.isDone()

    def place_order(self, symbol, action, stop_loss=None):
        contract = next(c for c in self.contracts if c.symbol == symbol)
        order = MarketOrder(action, 100)
        trade = self.ib.placeOrder(contract, order)
        self.pending_orders[symbol] = trade
        self.active_trades[symbol] = trade  # Track the active trade
        trade.orderStatusEvent += self.on_order_status_update
        print(f"Placed {action} order for {symbol}")

    def on_order_status_update(self, trade):
        symbol = trade.contract.symbol  # Get the symbol from the trade's contract
        if trade.orderStatus.status == 'Filled':
            if trade.order.action == 'BUY':
                self.positions[symbol] = 1
                print(f"Buy order for {symbol} filled")
                self.check_for_sell_signal(symbol)  # Check for sell signal after buy order is filled
            elif trade.order.action == 'SELL':
                self.positions[symbol] = 0
                print(f"Sell order for {symbol} filled")
            self.pending_orders[symbol] = None  # Clear the pending order once filled
            if trade.isDone():
                self.active_trades[symbol] = None  # Clear the active trade once done
            print(f"Order for {symbol} filled: {trade.order.action}")

    def check_for_sell_signal(self, symbol):
        df = self.combined_data[symbol]
        if df.empty or len(df) < 2:
            return
        
        ma_window = 50
        long_ma_window = 200
        
        stoch_rsi = StochRSIIndicator(close=df['close'], window=14, smooth1=3, smooth2=3, fillna=False)
        rsi = RSIIndicator(close=df['close'], window=14)
        bbands = BollingerBands(close=df['close'], window=20, window_dev=1)
        bbands2 = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['mavg'] = df['close'].rolling(window=ma_window, min_periods=1).mean()
        df['long_mavg'] = df['close'].rolling(window=long_ma_window, min_periods=1).mean()
        
        df['Mband'] = bbands.bollinger_mavg()
        df['Hband'] = bbands.bollinger_hband()
        df['Lband'] = bbands.bollinger_lband()
        df['Mband2'] = bbands2.bollinger_mavg()
        df['Hband2'] = bbands2.bollinger_hband()
        df['Lband2'] = bbands2.bollinger_lband()
        df['rsi'] = rsi.rsi()
        df['stochrsi'] = stoch_rsi.stochrsi()
        df['stoch%d'] = stoch_rsi.stochrsi_d()
        df['stoch%k'] = stoch_rsi.stochrsi_k()

        latest = df.iloc[-1]
        previous = df.iloc[-2]
        
        if self.active_trades[symbol] is None and self.positions[symbol] > 0:
            if self.trade_flags[symbol] == 'BANDS':
                if latest['close'] >= latest['Hband2']:
                    self.place_order(symbol, 'SELL')
                    self.positions[symbol] = 0
                    self.trade_flags[symbol] = None
            elif self.trade_flags[symbol] == 'TREND':
                if latest['mavg'] < latest['long_mavg'] or latest['rsi'] >= 70 or latest['stochrsi'] >= .80:
                    self.place_order(symbol, 'SELL')
                    self.positions[symbol] = 0
                    self.trade_flags[symbol] = None
            elif self.trade_flags[symbol] == 'RSI':
                if latest['close'] >= latest['Hband'] or latest['rsi'] >= 70:
                    self.place_order(symbol, 'SELL')
                    self.positions[symbol] = 0 
                    self.trade_flags[symbol] = None
            elif self.trade_flags[symbol] == 'STOCHRSI':
                if latest['mavg'] < latest['long_mavg'] or latest['stochrsi'] >= .80:
                    self.place_order(symbol, 'SELL')
                    self.positions[symbol] = 0
                    self.trade_flags[symbol] = None
                    
            
                 
    def run(self):
        self.qualify_contracts()
        self.fetch_historical_data()
        self.combine_data()
        self.fetch_live_data()
        print("Fetching data...")

if __name__ == "__main__":
    symbols = ['AAPL', 'GOOGL']
    bot = TradingBot(symbols=symbols)
    
    def websocket():
        bot.run()
        
    thread1 = threading.Thread(target=websocket(), daemon=True)
    thread1.start()

    # Keep the script running to fetch live data
    while True:
        time.sleep(5)
        for symbol in symbols:
            bot.aggregate_bars(symbol)
            bot.update_df(symbol)
