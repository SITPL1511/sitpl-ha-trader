import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, time as dt_time
import pytz
import logging
from typing import Dict, List, Tuple, Optional
import concurrent.futures
from dataclasses import dataclass
import warnings
import requests
import time
import random
from urllib.parse import urlencode
import json
warnings.filterwarnings('ignore')

@dataclass
class TradeSignal:
    """Trade signal data structure"""
    timestamp: datetime
    symbol: str
    signal_type: str  # 'LONG', 'SHORT', 'EXIT_LONG', 'EXIT_SHORT', 'EOD_EXIT'
    entry_price: float
    stop_loss: float
    ha_close: float
    ha_open: float
    regular_close: float
    regular_open: float

@dataclass
class Trade:
    """Trade execution data structure"""
    id: int
    symbol: str
    side: str  # 'LONG', 'SHORT'
    entry_time: datetime
    entry_price: float
    stop_loss: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    status: str = 'OPEN'  # 'OPEN', 'CLOSED'

class TradingConfig:
    """Trading engine configuration"""
    def __init__(self):
        self.symbols = ['AAPL', 'MSFT', 'GOOGL']
        self.timeframe = '5m'  # 1m, 5m, 15m, 1h, 1d
        self.position_size = 1000  # USD per trade
        self.max_positions = 5
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.data_lookback_days = 252
        self.price_tolerance = 0.001  # 0.1% for open=low/high detection
        self.concurrent_downloads = True

        # IST timezone configuration
        self.timezone = pytz.timezone('Asia/Kolkata')

        # EOD settings for different markets
        self.market_hours = {
            'NSE': {  # Indian market (NSE/BSE)
                'open': dt_time(9, 15),    # 9:15 AM IST
                'close': dt_time(15, 30),  # 3:30 PM IST
                'eod_exit_time': dt_time(15, 15)  # Close positions 15 minutes before market close
            },
            'NYSE': {  # US market (converted to IST)
                'open': dt_time(19, 30),   # 9:30 AM EST = 7:00 PM IST (winter) / 7:30 PM IST (summer)
                'close': dt_time(2, 0),    # 4:00 PM EST = 2:30 AM IST (next day)
                'eod_exit_time': dt_time(1, 45)  # Close positions 15 minutes before market close
            },
            'NASDAQ': {  # Same as NYSE
                'open': dt_time(19, 30),
                'close': dt_time(2, 0),
                'eod_exit_time': dt_time(1, 45)
            }
        }

        # Default market (will be auto-detected based on symbol)
        self.default_market = 'NSE'

        # Enable/disable EOD closure
        self.enable_eod_closure = True

        # API Keys for alternative data sources
        self.alpha_vantage_key = "GFAICALUQMRA3RBJ"
        self.polygon_key = "demo"

        # Data source preference order
        self.data_sources = ['alpha_vantage', 'polygon', 'yahoo_direct', 'yfinance']

    def detect_market(self, symbol: str) -> str:
        """Detect market based on symbol suffix"""
        if symbol.endswith('.NS') or symbol.endswith('.BO'):
            return 'NSE'
        elif any(symbol.endswith(suffix) for suffix in ['.L', '.TO', '.AX', '-USD']):
            return 'OTHER'
        else:
            return 'NSE'  # Default for US stocks

class TimeManager:
    """Handles timezone conversions and market hours logic"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        self.utc_tz = pytz.UTC
        self.logger = logging.getLogger(__name__)

    def convert_to_ist(self, timestamp) -> datetime:
        """Convert any timestamp to IST timezone"""
        if timestamp is None:
            return None

        # Handle different input types
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        elif isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()

        # If timezone-naive, assume UTC
        if timestamp.tzinfo is None:
            timestamp = self.utc_tz.localize(timestamp)

        # Convert to IST
        ist_time = timestamp.astimezone(self.ist_tz)
        return ist_time

    def is_market_open(self, timestamp: datetime, symbol: str) -> bool:
        """Check if market is open for given symbol at given time"""
        market = self.config.detect_market(symbol)

        if market not in self.config.market_hours:
            return True  # Default to always open for unknown markets

        market_config = self.config.market_hours[market]
        ist_time = self.convert_to_ist(timestamp)
        current_time = ist_time.time()

        # Handle markets that close next day (like NYSE)
        if market_config['close'] < market_config['open']:
            # Market spans midnight
            return current_time >= market_config['open'] or current_time <= market_config['close']
        else:
            # Regular market hours
            return market_config['open'] <= current_time <= market_config['close']

    def should_exit_eod(self, timestamp: datetime, symbol: str) -> bool:
        """Check if position should be closed due to EOD"""
        if not self.config.enable_eod_closure:
            return False

        market = self.config.detect_market(symbol)

        if market not in self.config.market_hours:
            return False

        market_config = self.config.market_hours[market]
        ist_time = self.convert_to_ist(timestamp)
        current_time = ist_time.time()

        # Check if current time is past EOD exit time
        eod_exit = market_config['eod_exit_time']

        if market_config['close'] < market_config['open']:
            # Market spans midnight - need special handling
            if eod_exit < market_config['open']:
                # EOD exit is next day
                return current_time >= eod_exit and current_time <= market_config['close']
            else:
                # EOD exit is same day
                return current_time >= eod_exit
        else:
            # Regular market hours
            return current_time >= eod_exit

    def get_next_eod_time(self, timestamp: datetime, symbol: str) -> datetime:
        """Get the next EOD exit time for a symbol"""
        market = self.config.detect_market(symbol)

        if market not in self.config.market_hours:
            # Default: assume next day 3:15 PM IST
            ist_time = self.convert_to_ist(timestamp)
            next_eod = ist_time.replace(hour=15, minute=15, second=0, microsecond=0)
            if next_eod <= ist_time:
                next_eod += timedelta(days=1)
            return next_eod

        market_config = self.config.market_hours[market]
        ist_time = self.convert_to_ist(timestamp)

        # Calculate next EOD exit time
        eod_exit = market_config['eod_exit_time']
        next_eod = ist_time.replace(
            hour=eod_exit.hour,
            minute=eod_exit.minute,
            second=0,
            microsecond=0
        )

        # If EOD time has passed today, move to next trading day
        if next_eod <= ist_time:
            next_eod += timedelta(days=1)

        return next_eod

class DataManager:
    """Handles data downloading from multiple sources with fallback options"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.time_manager = TimeManager(config)
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.setup_session()

    def setup_session(self):
        """Setup requests session with headers to avoid rate limiting"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        self.session.headers.update(headers)

    def convert_timestamps_to_ist(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert all timestamps in dataframe to IST"""
        if 'Datetime' in df.columns:
            df['Datetime'] = df['Datetime'].apply(self.time_manager.convert_to_ist)
        elif df.index.name == 'Datetime' or isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.DatetimeIndex([self.time_manager.convert_to_ist(ts) for ts in df.index])

        return df

    def download_data_alpha_vantage(self, symbol: str, api_key: str = "demo",
                                    interval: str = "5min", outputsize: str = "compact") -> Optional[pd.DataFrame]:
        """Download data from Alpha Vantage API with IST conversion"""
        try:
            base_url = "https://www.alphavantage.co/query"

            # Map intervals
            av_intervals = {
                '1m': '1min', '5m': '5min', '15m': '15min',
                '30m': '30min', '1h': '60min', '1d': 'daily'
            }
            av_interval = av_intervals.get(interval, '5min')

            if av_interval in ['1min', '5min', '15min', '30min', '60min']:
                function = "TIME_SERIES_INTRADAY"
                params = {
                    'function': function,
                    'symbol': symbol,
                    'interval': av_interval,
                    'apikey': api_key,
                    'outputsize': outputsize,
                    'datatype': 'json'
                }
            else:
                function = "TIME_SERIES_DAILY"
                params = {
                    'function': function,
                    'symbol': symbol,
                    'apikey': api_key,
                    'outputsize': outputsize,
                    'datatype': 'json'
                }

            response = self.session.get(base_url, params=params)
            response.raise_for_status()

            data = response.json()

            # Check for API errors
            if 'Error Message' in data:
                self.logger.error(f"Alpha Vantage error for {symbol}: {data['Error Message']}")
                return None

            if 'Note' in data:
                self.logger.warning(f"Alpha Vantage rate limit for {symbol}: {data['Note']}")
                return None

            # Extract time series data
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break

            if not time_series_key:
                self.logger.error(f"No time series data found for {symbol}")
                return None

            time_series = data[time_series_key]

            # Convert to DataFrame
            df_data = []
            for timestamp, values in time_series.items():
                df_data.append({
                    'Datetime': pd.to_datetime(timestamp),
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Volume': int(values['5. volume'])
                })

            df = pd.DataFrame(df_data)
            df = df.sort_values('Datetime').reset_index(drop=True)

            # Convert to IST
            df = self.convert_timestamps_to_ist(df)

            self.logger.info(f"Downloaded {len(df)} bars for {symbol} from Alpha Vantage (converted to IST)")
            return df

        except Exception as e:
            self.logger.error(f"Alpha Vantage error for {symbol}: {e}")
            return None

    def download_data_polygon(self, symbol: str, api_key: str = "demo",
                              multiplier: int = 5, timespan: str = "minute",
                              from_date: str = None, to_date: str = None) -> Optional[pd.DataFrame]:
        """Download data from Polygon.io API with IST conversion"""
        try:
            if not from_date:
                from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            if not to_date:
                to_date = datetime.now().strftime('%Y-%m-%d')

            base_url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from_date}/{to_date}"

            params = {
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000,
                'apikey': api_key
            }

            response = self.session.get(base_url, params=params)
            response.raise_for_status()

            data = response.json()

            if data['status'] != 'OK':
                self.logger.error(f"Polygon error for {symbol}: {data.get('error', 'Unknown error')}")
                return None

            if 'results' not in data or not data['results']:
                self.logger.warning(f"No data found for {symbol} on Polygon")
                return None

            # Convert to DataFrame
            df_data = []
            for bar in data['results']:
                df_data.append({
                    'Datetime': pd.to_datetime(bar['t'], unit='ms'),
                    'Open': bar['o'],
                    'High': bar['h'],
                    'Low': bar['l'],
                    'Close': bar['c'],
                    'Volume': bar['v']
                })

            df = pd.DataFrame(df_data)
            df = df.sort_values('Datetime').reset_index(drop=True)

            # Convert to IST
            df = self.convert_timestamps_to_ist(df)

            self.logger.info(f"Downloaded {len(df)} bars for {symbol} from Polygon (converted to IST)")
            return df

        except Exception as e:
            self.logger.error(f"Polygon error for {symbol}: {e}")
            return None

    def download_data_yahoo_direct(self, symbol: str, period1: int = None,
                                   period2: int = None, interval: str = "5m") -> Optional[pd.DataFrame]:
        """Direct Yahoo Finance API access with IST conversion"""
        try:
            if not period1:
                period1 = int((datetime.now() - timedelta(days=30)).timestamp())
            if not period2:
                period2 = int(datetime.now().timestamp())

            # Add random delay to avoid rate limiting
            time.sleep(random.uniform(0.1, 0.5))

            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                'period1': period1,
                'period2': period2,
                'interval': interval,
                'includePrePost': 'false',
                'events': 'div|split'
            }

            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 429:
                self.logger.warning(f"Rate limited for {symbol}, waiting...")
                time.sleep(random.uniform(5, 10))
                return None

            response.raise_for_status()
            data = response.json()

            if 'chart' not in data or not data['chart']['result']:
                self.logger.error(f"No data in Yahoo response for {symbol}")
                return None

            result = data['chart']['result'][0]

            if 'timestamp' not in result:
                self.logger.error(f"No timestamp data for {symbol}")
                return None

            timestamps = result['timestamp']
            indicators = result['indicators']['quote'][0]

            df_data = []
            for i, ts in enumerate(timestamps):
                if (indicators['open'][i] is not None and
                        indicators['high'][i] is not None and
                        indicators['low'][i] is not None and
                        indicators['close'][i] is not None):

                    df_data.append({
                        'Datetime': pd.to_datetime(ts, unit='s'),
                        'Open': indicators['open'][i],
                        'High': indicators['high'][i],
                        'Low': indicators['low'][i],
                        'Close': indicators['close'][i],
                        'Volume': indicators['volume'][i] if indicators['volume'][i] else 0
                    })

            if not df_data:
                self.logger.error(f"No valid OHLC data for {symbol}")
                return None

            df = pd.DataFrame(df_data)
            df = df.sort_values('Datetime').reset_index(drop=True)

            # Convert to IST
            df = self.convert_timestamps_to_ist(df)

            self.logger.info(f"Downloaded {len(df)} bars for {symbol} from Yahoo Direct (converted to IST)")
            return df

        except Exception as e:
            self.logger.error(f"Yahoo Direct error for {symbol}: {e}")
            return None

    def download_data_with_fallback(self, symbol: str, **kwargs) -> Optional[pd.DataFrame]:
        """Try multiple data sources with fallback logic"""
        # Method 1: Alpha Vantage (if API key provided)
        alpha_vantage_key = kwargs.get('alpha_vantage_key')
        if alpha_vantage_key and alpha_vantage_key != "demo":
            self.logger.info(f"Trying Alpha Vantage for {symbol}")
            data = self.download_data_alpha_vantage(symbol, alpha_vantage_key,
                                                    kwargs.get('interval', '5m'))
            if data is not None:
                return data
            time.sleep(1)

        # Method 2: Polygon (if API key provided)
        polygon_key = kwargs.get('polygon_key')
        if polygon_key and polygon_key != "demo":
            self.logger.info(f"Trying Polygon for {symbol}")
            data = self.download_data_polygon(symbol, polygon_key,
                                              timespan=kwargs.get('timespan', 'minute'))
            if data is not None:
                return data
            time.sleep(1)

        # Method 3: Yahoo Direct API
        self.logger.info(f"Trying Yahoo Direct API for {symbol}")
        data = self.download_data_yahoo_direct(symbol, interval=kwargs.get('interval', '5m'))
        if data is not None:
            return data

        # Method 4: yfinance as last resort
        self.logger.info(f"Trying yfinance as fallback for {symbol}")
        data = self._download_yfinance_with_retry(symbol, **kwargs)
        if data is not None:
            return data

        self.logger.error(f"All data sources failed for {symbol}")
        return None

    def _download_yfinance_with_retry(self, symbol: str, max_retries: int = 3, **kwargs) -> Optional[pd.DataFrame]:
        """yfinance with exponential backoff retry logic and IST conversion"""
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    self.logger.info(f"Retrying {symbol} after {delay:.1f}s delay")
                    time.sleep(delay)

                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    period=kwargs.get('period', '1y'),
                    interval=kwargs.get('interval', '5m')
                )

                if not data.empty:
                    data.reset_index(inplace=True)
                    # Convert to IST
                    data = self.convert_timestamps_to_ist(data)
                    self.logger.info(f"Downloaded {len(data)} bars for {symbol} from yfinance (converted to IST)")
                    return data

            except Exception as e:
                error_msg = str(e).lower()
                if '429' in error_msg or 'rate limit' in error_msg:
                    self.logger.warning(f"Rate limited for {symbol}, attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        continue
                else:
                    self.logger.error(f"yfinance error for {symbol}: {e}")
                    break

        return None

    def validate_symbols(self, symbols: List[str]) -> List[str]:
        """Validate symbols and filter out delisted/invalid ones"""
        valid_symbols = []

        self.logger.info("Validating symbols...")

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="5d", interval="1d")

                if not data.empty and len(data) >= 1:
                    last_date = data.index[-1].date() if hasattr(data.index[-1], 'date') else data.index[-1]
                    days_old = (datetime.now().date() - last_date).days

                    if days_old <= 7:
                        valid_symbols.append(symbol)
                        self.logger.info(f"✓ {symbol} is valid (last data: {days_old} days ago)")
                    else:
                        self.logger.warning(f"✗ {symbol} may be delisted (last data: {days_old} days ago)")
                else:
                    self.logger.warning(f"✗ {symbol} has no recent data - likely delisted")

            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['delisted', 'not found', 'invalid', '404']):
                    self.logger.error(f"✗ {symbol} is delisted or invalid: {e}")
                else:
                    self.logger.error(f"✗ Error validating {symbol}: {e}")

        self.logger.info(f"Validation complete: {len(valid_symbols)}/{len(symbols)} symbols are valid")
        return valid_symbols

    def download_data(self, symbols: List[str], period: str = "1y",
                      interval: str = "5m", **kwargs) -> Dict[str, pd.DataFrame]:
        """Download historical data for multiple symbols with multiple data sources and IST conversion"""
        if not kwargs.get('skip_validation', False):
            valid_symbols = self.validate_symbols(symbols)
        else:
            valid_symbols = symbols

        if not valid_symbols:
            self.logger.error("No valid symbols found!")
            return {}

        if self.config.concurrent_downloads:
            return self._download_concurrent_fallback(valid_symbols, period, interval, **kwargs)
        else:
            return self._download_sequential_fallback(valid_symbols, period, interval, **kwargs)

    def _download_concurrent_fallback(self, symbols: List[str], period: str,
                                      interval: str, **kwargs) -> Dict[str, pd.DataFrame]:
        """Download using multiple sources concurrently"""
        def download_single_fallback(symbol):
            return symbol, self.download_data_with_fallback(
                symbol, period=period, interval=interval, **kwargs
            )

        results = {}
        max_workers = min(3, len(symbols))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(download_single_fallback, symbol): symbol
                for symbol in symbols
            }

            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol, data = future.result()
                if data is not None:
                    results[symbol] = data
                time.sleep(0.5)

        return results

    def _download_sequential_fallback(self, symbols: List[str], period: str,
                                      interval: str, **kwargs) -> Dict[str, pd.DataFrame]:
        """Download using fallback sources sequentially"""
        results = {}

        for i, symbol in enumerate(symbols):
            self.logger.info(f"Processing {symbol} ({i+1}/{len(symbols)})")

            data = self.download_data_with_fallback(
                symbol, period=period, interval=interval, **kwargs
            )

            if data is not None:
                results[symbol] = data

            if i < len(symbols) - 1:
                delay = random.uniform(1, 3)
                time.sleep(delay)

        return results

class HeikinAshiCalculator:
    """High-performance Heikin Ashi calculations using vectorized operations"""

    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Heikin Ashi values using vectorized operations"""
        data = df.copy()
        n = len(data)

        # Pre-allocate arrays for performance
        ha_close = np.zeros(n)
        ha_open = np.zeros(n)
        ha_high = np.zeros(n)
        ha_low = np.zeros(n)

        # Vectorized HA_Close calculation
        ha_close = (data['Open'] + data['High'] + data['Low'] + data['Close']) / 4

        # HA_Open calculation (requires loop for dependency)
        ha_open[0] = (data['Open'].iloc[0] + data['Close'].iloc[0]) / 2
        for i in range(1, n):
            ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2

        # Vectorized HA_High and HA_Low
        ha_high = np.maximum.reduce([data['High'], ha_open, ha_close])
        ha_low = np.minimum.reduce([data['Low'], ha_open, ha_close])

        # Add to dataframe
        data['HA_Open'] = ha_open
        data['HA_High'] = ha_high
        data['HA_Low'] = ha_low
        data['HA_Close'] = ha_close
        data['HA_Color'] = np.where(ha_close >= ha_open, 'GREEN', 'RED')

        return data

class SignalGenerator:
    """Generates trading signals based on Heikin Ashi patterns with EOD logic"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.time_manager = TimeManager(config)
        self.logger = logging.getLogger(__name__)

    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[TradeSignal]:
        """Generate trading signals from Heikin Ashi data with EOD exits"""
        signals = []

        if len(data) < 2:
            return signals

        # Vectorized signal detection
        prev_ha_color = data['HA_Color'].shift(1)
        curr_ha_color = data['HA_Color']

        # Long entry conditions (all on HA candles)
        long_cond1 = prev_ha_color == 'GREEN'
        long_cond2 = data['HA_Close'] > data['HA_Close'].shift(1)
        long_cond3 = np.abs(data['HA_Open'] - data['HA_Low']) < (data['HA_High'] - data['HA_Low']) * self.config.price_tolerance

        long_entries = long_cond1 & long_cond2 & long_cond3

        # Short entry conditions (all on HA candles)
        short_cond1 = prev_ha_color == 'RED'
        short_cond2 = np.abs(data['HA_Open'] - data['HA_High']) < (data['HA_High'] - data['HA_Low']) * self.config.price_tolerance
        short_cond3 = data['HA_Close'] < data['HA_Close'].shift(1)

        short_entries = short_cond1 & short_cond2 & short_cond3

        # Exit conditions
        green_to_red = (prev_ha_color == 'GREEN') & (curr_ha_color == 'RED')
        red_to_green = (prev_ha_color == 'RED') & (curr_ha_color == 'GREEN')

        # Generate signals
        for i in range(1, len(data)):
            timestamp = data.iloc[i]['Datetime'] if 'Datetime' in data.columns else data.index[i]

            # Only generate entry signals during market hours
            if self.time_manager.is_market_open(timestamp, symbol):

                # Check if we should avoid entries near EOD
                if not self.time_manager.should_exit_eod(timestamp, symbol):

                    if long_entries.iloc[i]:
                        signals.append(TradeSignal(
                            timestamp=timestamp,
                            symbol=symbol,
                            signal_type='LONG',
                            entry_price=round(data.iloc[i]['Close'], 4),
                            stop_loss=min(round(data.iloc[i]['Open'], 4), round(data.iloc[i]['HA_Open'], 4)),
                            ha_close=round(data.iloc[i]['HA_Close'], 4),
                            ha_open=round(data.iloc[i]['HA_Open'], 4),
                            regular_close=round(data.iloc[i]['Close'], 4),
                            regular_open=round(data.iloc[i]['Open'], 4)
                        ))

                    if short_entries.iloc[i]:
                        signals.append(TradeSignal(
                            timestamp=timestamp,
                            symbol=symbol,
                            signal_type='SHORT',
                            entry_price=round(data.iloc[i]['Close'], 4),
                            stop_loss=max(round(data.iloc[i]['Open'], 4), round(data.iloc[i]['HA_Open'], 4)),
                            ha_close=round(data.iloc[i]['HA_Close'], 4),
                            ha_open=round(data.iloc[i]['HA_Open'], 4),
                            regular_close=round(data.iloc[i]['Close'], 4),
                            regular_open=round(data.iloc[i]['Open'], 4)
                        ))

            # Regular exit signals (HA color change)
            if green_to_red.iloc[i]:
                signals.append(TradeSignal(
                    timestamp=timestamp,
                    symbol=symbol,
                    signal_type='EXIT_LONG',
                    entry_price=round(data.iloc[i]['Close'], 4),
                    stop_loss=0.0000,
                    ha_close=round(data.iloc[i]['HA_Close'], 4),
                    ha_open=round(data.iloc[i]['HA_Open'], 4),
                    regular_close=round(data.iloc[i]['Close'], 4),
                    regular_open=round(data.iloc[i]['Open'], 4)
                ))

            if red_to_green.iloc[i]:
                signals.append(TradeSignal(
                    timestamp=timestamp,
                    symbol=symbol,
                    signal_type='EXIT_SHORT',
                    entry_price=round(data.iloc[i]['Close'], 4),
                    stop_loss=0.0000,
                    ha_close=round(data.iloc[i]['HA_Close'], 4),
                    ha_open=round(data.iloc[i]['HA_Open'], 4),
                    regular_close=round(data.iloc[i]['Close'], 4),
                    regular_open=round(data.iloc[i]['Open'], 4)
                ))

            # EOD exit signals
            if self.time_manager.should_exit_eod(timestamp, symbol):
                # Generate both long and short exit signals at EOD
                signals.append(TradeSignal(
                    timestamp=timestamp,
                    symbol=symbol,
                    signal_type='EOD_EXIT',
                    entry_price=round(data.iloc[i]['Close'], 4),
                    stop_loss=0.0000,
                    ha_close=round(data.iloc[i]['HA_Close'], 4),
                    ha_open=round(data.iloc[i]['HA_Open'], 4),
                    regular_close=round(data.iloc[i]['Close'], 4),
                    regular_open=round(data.iloc[i]['Open'], 4)
                ))

        return signals

class TradingEngine:
    """Main trading engine that orchestrates all components with IST and EOD logic"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.data_manager = DataManager(config)
        self.signal_generator = SignalGenerator(config)
        self.time_manager = TimeManager(config)
        self.trades: List[Trade] = []
        self.open_positions: Dict[str, Trade] = {}
        self.trade_id_counter = 0

        # EOD tracking
        self.eod_closures = []

        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)

    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'trading_engine_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )

    def run_backtest(self, start_date: str = None, end_date: str = None) -> Dict:
        """Run historical backtest with IST conversion and EOD logic"""
        start_time = datetime.now()
        self.logger.info("Starting backtest with IST timezone and EOD closure logic...")

        # Download data with API keys
        download_kwargs = {
            'alpha_vantage_key': self.config.alpha_vantage_key,
            'polygon_key': self.config.polygon_key,
            'skip_validation': True
        }

        data_dict = self.data_manager.download_data(
            symbols=self.config.symbols,
            period="1y",
            interval=self.config.timeframe,
            **download_kwargs
        )

        if not data_dict:
            self.logger.error("No data downloaded! Check your API keys or symbols.")
            return {'trades': [], 'performance': {'total_trades': 0}, 'execution_time': 0, 'total_signals': 0}

        # Process each symbol
        all_signals = []
        for symbol, data in data_dict.items():
            self.logger.info(f"Processing {symbol} - Market: {self.config.detect_market(symbol)}")

            # Calculate Heikin Ashi
            ha_data = HeikinAshiCalculator.calculate(data)

            # Generate signals with symbol-specific logic
            signals = self.signal_generator.generate_signals(ha_data, symbol)
            all_signals.extend(signals)

            self.logger.info(f"Generated {len(signals)} signals for {symbol}")

        # Sort signals by timestamp (IST)
        all_signals.sort(key=lambda x: x.timestamp)

        self.logger.info(f"Total signals generated: {len(all_signals)}")

        # Execute trades
        for signal in all_signals:
            self._process_signal(signal)

        # Force close any remaining open positions at the end
        self._close_all_positions_at_end()

        # Calculate performance metrics
        performance = self._calculate_performance()

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        self.logger.info(f"Backtest completed in {execution_time:.2f} seconds")
        self.logger.info(f"Total trades: {len(self.trades)}")
        self.logger.info(f"EOD closures: {len(self.eod_closures)}")

        return {
            'trades': self.trades,
            'performance': performance,
            'execution_time': execution_time,
            'total_signals': len(all_signals),
            'data_sources_used': list(data_dict.keys()),
            'eod_closures': len(self.eod_closures),
            'timezone': 'Asia/Kolkata (IST)'
        }

    def _process_signal(self, signal: TradeSignal):
        """Process individual trading signal with EOD logic"""
        symbol = signal.symbol

        if signal.signal_type in ['LONG', 'SHORT']:
            # Check if we already have a position
            if symbol in self.open_positions:
                return

            # Check position limits
            if len(self.open_positions) >= self.config.max_positions:
                return

            # Create new trade
            trade = Trade(
                id=self.trade_id_counter,
                symbol=symbol,
                side=signal.signal_type,
                entry_time=signal.timestamp,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss
            )

            self.trades.append(trade)
            self.open_positions[symbol] = trade
            self.trade_id_counter += 1

            ist_time = signal.timestamp.strftime('%Y-%m-%d %H:%M:%S IST')
            self.logger.info(f"Opened {signal.signal_type} position for {symbol} at {round(signal.entry_price, 4)} [{ist_time}]")

        elif signal.signal_type in ['EXIT_LONG', 'EXIT_SHORT']:
            if symbol in self.open_positions:
                trade = self.open_positions[symbol]

                # Check if exit matches position type
                if (signal.signal_type == 'EXIT_LONG' and trade.side == 'LONG') or \
                        (signal.signal_type == 'EXIT_SHORT' and trade.side == 'SHORT'):

                    self._close_position(trade, signal, 'HA_COLOR_CHANGE')

        elif signal.signal_type == 'EOD_EXIT':
            # Close any open position for this symbol at EOD
            if symbol in self.open_positions:
                trade = self.open_positions[symbol]
                self._close_position(trade, signal, 'EOD_CLOSURE')
                self.eod_closures.append({
                    'symbol': symbol,
                    'timestamp': signal.timestamp,
                    'side': trade.side
                })

    def _close_position(self, trade: Trade, signal: TradeSignal, reason: str):
        """Close a position and calculate P&L"""
        trade.exit_time = signal.timestamp
        trade.exit_price = round(signal.entry_price, 4)  # Using regular chart price
        trade.exit_reason = reason
        trade.status = 'CLOSED'

        # Calculate P&L
        if trade.side == 'LONG':
            trade.pnl = round(trade.exit_price - trade.entry_price, 4)
        else:  # SHORT
            trade.pnl = round(trade.entry_price - trade.exit_price, 4)

        del self.open_positions[trade.symbol]

        ist_time = signal.timestamp.strftime('%Y-%m-%d %H:%M:%S IST')
        self.logger.info(f"Closed {trade.side} position for {trade.symbol} at {trade.exit_price:.4f}, P&L: {trade.pnl:.4f}, Reason: {reason} [{ist_time}]")

    def _close_all_positions_at_end(self):
        """Force close all remaining open positions at the end of backtest"""
        if self.open_positions:
            self.logger.info(f"Force closing {len(self.open_positions)} remaining open positions")

            for symbol, trade in list(self.open_positions.items()):
                # Use last known price as exit price
                trade.exit_time = datetime.now(self.time_manager.ist_tz)
                trade.exit_price = round(trade.entry_price, 4)  # Conservative estimate
                trade.exit_reason = 'BACKTEST_END'
                trade.status = 'CLOSED'

                # Calculate P&L (assuming no change for safety)
                trade.pnl = 0.0000

                self.logger.info(f"Force closed {trade.side} position for {symbol} at backtest end")

            self.open_positions.clear()

    def _calculate_performance(self) -> Dict:
        """Calculate performance metrics with IST timezone info"""
        closed_trades = [t for t in self.trades if t.status == 'CLOSED']

        if not closed_trades:
            return {'total_trades': 0, 'timezone': 'Asia/Kolkata (IST)'}

        pnls = [t.pnl for t in closed_trades if t.pnl is not None]

        if not pnls:
            return {'total_trades': len(closed_trades), 'timezone': 'Asia/Kolkata (IST)'}

        total_pnl = round(sum(pnls), 4)
        win_trades = [p for p in pnls if p > 0]
        lose_trades = [p for p in pnls if p < 0]

        # Calculate trade durations
        durations = []
        eod_trades = 0
        ha_exit_trades = 0

        for trade in closed_trades:
            if trade.entry_time and trade.exit_time:
                duration = trade.exit_time - trade.entry_time
                durations.append(duration.total_seconds() / 3600)  # Convert to hours

                if trade.exit_reason == 'EOD_CLOSURE':
                    eod_trades += 1
                elif trade.exit_reason == 'HA_COLOR_CHANGE':
                    ha_exit_trades += 1

        metrics = {
            'total_trades': len(closed_trades),
            'winning_trades': len(win_trades),
            'losing_trades': len(lose_trades),
            'win_rate': round(len(win_trades) / len(closed_trades), 4) if closed_trades else 0.0000,
            'total_pnl': total_pnl,
            'average_win': round(np.mean(win_trades), 4) if win_trades else 0.0000,
            'average_loss': round(np.mean(lose_trades), 4) if lose_trades else 0.0000,
            'profit_factor': round(abs(sum(win_trades) / sum(lose_trades)), 4) if lose_trades and sum(lose_trades) != 0 else float('inf'),
            'max_win': round(max(pnls), 4) if pnls else 0.0000,
            'max_loss': round(min(pnls), 4) if pnls else 0.0000,
            'average_trade_duration_hours': round(np.mean(durations), 4) if durations else 0.0000,
            'eod_closures': eod_trades,
            'ha_color_exits': ha_exit_trades,
            'timezone': 'Asia/Kolkata (IST)'
        }

        # Calculate drawdown
        cumulative_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        metrics['max_drawdown'] = round(np.max(drawdown), 4) if len(drawdown) > 0 else 0.0000

        return metrics

    def export_trades(self, filename: str = None):
        """Export trades to CSV with IST timestamps"""
        if not filename:
            filename = f"trades_ist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        trades_data = []
        for trade in self.trades:
            # Format IST timestamps
            entry_time_ist = trade.entry_time.strftime('%Y-%m-%d %H:%M:%S IST') if trade.entry_time else None
            exit_time_ist = trade.exit_time.strftime('%Y-%m-%d %H:%M:%S IST') if trade.exit_time else None

            trades_data.append({
                'id': trade.id,
                'symbol': trade.symbol,
                'market': self.config.detect_market(trade.symbol),
                'side': trade.side,
                'entry_time_ist': entry_time_ist,
                'entry_price': round(trade.entry_price, 4) if trade.entry_price else None,
                'stop_loss': round(trade.stop_loss, 4) if trade.stop_loss else None,
                'exit_time_ist': exit_time_ist,
                'exit_price': round(trade.exit_price, 4) if trade.exit_price else None,
                'exit_reason': trade.exit_reason,
                'pnl': round(trade.pnl, 4) if trade.pnl is not None else None,
                'status': trade.status
            })

        df = pd.DataFrame(trades_data)
        df.to_csv(filename, index=False)
        self.logger.info(f"Trades exported to {filename} with IST timestamps")

    def print_market_hours_info(self):
        """Print configured market hours for reference"""
        print("\n=== CONFIGURED MARKET HOURS (IST) ===")
        for market, hours in self.config.market_hours.items():
            print(f"{market}:")
            print(f"  Open: {hours['open'].strftime('%H:%M')}")
            print(f"  Close: {hours['close'].strftime('%H:%M')}")
            print(f"  EOD Exit: {hours['eod_exit_time'].strftime('%H:%M')}")
            print()

def main():
    """Main execution function with IST and EOD enhancements"""
    # Initialize configuration
    config = TradingConfig()

    # Set your API keys here
    config.alpha_vantage_key = "demo"  # Replace with your free Alpha Vantage API key
    config.polygon_key = "demo"        # Replace with your free Polygon API key

    # Configure for Indian market
    config.symbols = ['MAZDOCK.NS']  # Indian stock symbol
    config.timeframe = '15m'
    config.concurrent_downloads = False
    config.enable_eod_closure = True  # Enable EOD position closure

    print("=== ENHANCED TRADING ENGINE WITH IST & EOD LOGIC ===")
    print("Features:")
    print("✓ Multi-source data downloader with fallback")
    print("✓ Automatic timezone conversion to IST")
    print("✓ Market hours detection and validation")
    print("✓ End-of-Day position closure to avoid gap risks")
    print("✓ Separate tracking of EOD vs HA-based exits")
    print()

    # Initialize trading engine
    engine = TradingEngine(config)

    # Print market hours configuration
    engine.print_market_hours_info()

    try:
        # Run backtest
        results = engine.run_backtest()

        # Print performance summary
        performance = results['performance']
        print("\n=== BACKTEST RESULTS ===")
        print(f"Timezone: {results.get('timezone', 'UTC')}")
        print(f"Data Sources Used: {results.get('data_sources_used', [])}")
        print(f"Execution Time: {results['execution_time']:.2f} seconds")
        print(f"Total Signals: {results['total_signals']}")
        print(f"Total Trades: {performance.get('total_trades', 0)}")

        if performance.get('total_trades', 0) > 0:
            print(f"Win Rate: {performance.get('win_rate', 0):.4f} ({performance.get('win_rate', 0)*100:.2f}%)")
            print(f"Total P&L: ${performance.get('total_pnl', 0):.4f}")
            print(f"Max Drawdown: ${performance.get('max_drawdown', 0):.4f}")
            print(f"Profit Factor: {performance.get('profit_factor', 0):.4f}")
            print(f"Average Win: ${performance.get('average_win', 0):.4f}")
            print(f"Average Loss: ${performance.get('average_loss', 0):.4f}")
            print(f"Average Trade Duration: {performance.get('average_trade_duration_hours', 0):.4f} hours")
            print(f"EOD Closures: {performance.get('eod_closures', 0)}")
            print(f"HA Color Change Exits: {performance.get('ha_color_exits', 0)}")
        else:
            print("No completed trades found.")

        # Export trades if any exist
        if len(engine.trades) > 0:
            engine.export_trades()
            print(f"\nTrade history exported to CSV with IST timestamps")

        print("\n=== EOD RISK MANAGEMENT ===")
        print("✓ Positions automatically closed before market close")
        print("✓ Prevents overnight gap risk exposure")
        print("✓ Configurable EOD exit time per market")
        print("✓ Separate tracking of EOD vs signal-based exits")

        return engine

    except Exception as e:
        print(f"Error during backtest: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Performance test
    import time
    start_time = time.time()

    engine = main()

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    print("Enhanced trading engine completed successfully!")