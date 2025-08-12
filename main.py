import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
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
import pytz
warnings.filterwarnings('ignore')

@dataclass
class TradeSignal:
    """Trade signal data structure"""
    timestamp: datetime
    symbol: str
    signal_type: str  # 'LONG', 'SHORT', 'EXIT_LONG', 'EXIT_SHORT'
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

        # API Keys for alternative data sources
        self.alpha_vantage_key = "demo"  # Get free key from https://www.alphavantage.co/support/#api-key
        self.polygon_key = "demo"        # Get free key from https://polygon.io/

        # Data source preference order
        self.data_sources = ['alpha_vantage', 'polygon', 'yahoo_direct', 'yfinance']

        # Timezone configuration
        self.timezone = 'Asia/Kolkata'  # IST timezone
        self.ist_tz = pytz.timezone(self.timezone)

class DataManager:
    """Handles data downloading from multiple sources with fallback options"""

    def __init__(self, config: TradingConfig):
        self.config = config
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

    def convert_to_ist(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame timestamps to IST timezone"""
        df = df.copy()

        if 'Datetime' not in df.columns:
            return df

        # Ensure the datetime column is datetime type
        df['Datetime'] = pd.to_datetime(df['Datetime'])

        # If timezone-naive, assume UTC
        if df['Datetime'].dt.tz is None:
            df['Datetime'] = df['Datetime'].dt.tz_localize('UTC')

        # Convert to IST
        df['Datetime'] = df['Datetime'].dt.tz_convert(self.config.ist_tz)

        # Add IST offset info for clarity
        df['IST_Offset'] = '+05:30'

        self.logger.info(f"Converted timestamps to IST timezone ({self.config.timezone})")
        return df

    def download_data_alpha_vantage(self, symbol: str, api_key: str = "demo",
                                    interval: str = "5min", outputsize: str = "compact") -> Optional[pd.DataFrame]:
        """
        Download data from Alpha Vantage API (free tier: 5 calls/minute, 500 calls/day)
        Get free API key from: https://www.alphavantage.co/support/#api-key
        """
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

    def convert_to_ist_yfinance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert yfinance DataFrame to IST timezone (handles different column structure)"""
        df = df.copy()

        # yfinance uses different column names
        datetime_col = None
        if 'Datetime' in df.columns:
            datetime_col = 'Datetime'
        elif 'Date' in df.columns:
            datetime_col = 'Date'
        elif df.index.name == 'Date' or df.index.name == 'Datetime':
            df.reset_index(inplace=True)
            datetime_col = df.columns[0]

        if datetime_col is None:
            self.logger.warning("No datetime column found in yfinance data")
            return df

        # Ensure the datetime column is datetime type
        df[datetime_col] = pd.to_datetime(df[datetime_col])

        # If timezone-naive, assume UTC
        if df[datetime_col].dt.tz is None:
            df[datetime_col] = df[datetime_col].dt.tz_localize('UTC')

        # Convert to IST
        df[datetime_col] = df[datetime_col].dt.tz_convert(self.config.ist_tz)

        # Rename to standard column name
        if datetime_col != 'Datetime':
            df = df.rename(columns={datetime_col: 'Datetime'})

        # Add IST offset info
        df['IST_Offset'] = '+05:30'

        self.logger.info(f"Converted yfinance timestamps to IST timezone")
        return df

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

        # Convert to IST timezone
        df = self.convert_to_ist(df)

        self.logger.info(f"Downloaded {len(df)} bars for {symbol} from Alpha Vantage")
        return df

    except Exception as e:
    self.logger.error(f"Alpha Vantage error for {symbol}: {e}")
    return None

def download_data_polygon(self, symbol: str, api_key: str = "demo",
                          multiplier: int = 5, timespan: str = "minute",
                          from_date: str = None, to_date: str = None) -> Optional[pd.DataFrame]:
    """
    Download data from Polygon.io API (free tier: 5 calls/minute)
    Get free API key from: https://polygon.io/
    """
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

        # Convert to IST timezone
        df = self.convert_to_ist(df)

        self.logger.info(f"Downloaded {len(df)} bars for {symbol} from Polygon")
        return df

    except Exception as e:
        self.logger.error(f"Polygon error for {symbol}: {e}")
        return None

def download_data_yahoo_direct(self, symbol: str, period1: int = None,
                               period2: int = None, interval: str = "5m") -> Optional[pd.DataFrame]:
    """
    Direct Yahoo Finance API access with better rate limiting handling
    """
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

        # Convert to IST timezone
        df = self.convert_to_ist(df)

        self.logger.info(f"Downloaded {len(df)} bars for {symbol} from Yahoo Direct")
        return df

    except Exception as e:
        self.logger.error(f"Yahoo Direct error for {symbol}: {e}")
        return None

def download_data_with_fallback(self, symbol: str, **kwargs) -> Optional[pd.DataFrame]:
    """
    Try multiple data sources with fallback logic
    Priority: Alpha Vantage -> Polygon -> Yahoo Direct -> yfinance
    """
    # Method 1: Alpha Vantage (if API key provided)
    alpha_vantage_key = kwargs.get('alpha_vantage_key')
    if alpha_vantage_key and alpha_vantage_key != "demo":
        self.logger.info(f"Trying Alpha Vantage for {symbol}")
        data = self.download_data_alpha_vantage(symbol, alpha_vantage_key,
                                                kwargs.get('interval', '5m'))
        if data is not None:
            return data
        time.sleep(1)  # Rate limiting

    # Method 2: Polygon (if API key provided)
    polygon_key = kwargs.get('polygon_key')
    if polygon_key and polygon_key != "demo":
        self.logger.info(f"Trying Polygon for {symbol}")
        data = self.download_data_polygon(symbol, polygon_key,
                                          timespan=kwargs.get('timespan', 'minute'))
        if data is not None:
            return data
        time.sleep(1)  # Rate limiting

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
    """yfinance with exponential backoff retry logic"""
    for attempt in range(max_retries):
        try:
            # Progressive delay to avoid rate limiting
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
                # Convert to IST timezone
                data = self.convert_to_ist_yfinance(data)
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

            # Quick validation with recent data
            data = ticker.history(period="5d", interval="1d")

            if not data.empty and len(data) >= 1:
                # Get current time in IST for comparison
                now_ist = datetime.now(self.config.ist_tz)

                # Check if data is recent (within last 7 days)
                last_date = data.index[-1]
                if last_date.tz is None:
                    last_date = last_date.tz_localize('UTC')
                last_date_ist = last_date.tz_convert(self.config.ist_tz)

                days_old = (now_ist.date() - last_date_ist.date()).days

                if days_old <= 7:  # Recent data available
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
    """Download historical data for multiple symbols with multiple data sources"""
    # First validate symbols (only if using yfinance validation)
    if not kwargs.get('skip_validation', False):
        valid_symbols = self.validate_symbols(symbols)
    else:
        valid_symbols = symbols

    if not valid_symbols:
        self.logger.error("No valid symbols found!")
        return {}

    # Use fallback method for better reliability
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
    # Limit concurrent requests to avoid overwhelming APIs
    max_workers = min(3, len(symbols))  # Reduced from 10 to 3

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(download_single_fallback, symbol): symbol
            for symbol in symbols
        }

        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol, data = future.result()
            if data is not None:
                results[symbol] = data

            # Add delay between requests
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

        # Add progressive delay to avoid rate limiting
        if i < len(symbols) - 1:  # Don't delay after last symbol
            delay = random.uniform(1, 3)  # Random delay between 1-3 seconds
            time.sleep(delay)

    return results

def _download_concurrent(self, symbols: List[str], period: str,
                         interval: str) -> Dict[str, pd.DataFrame]:
    """Download data concurrently for better performance"""
    def download_single(symbol):
        try:
            ticker = yf.Ticker(symbol)

            # Get basic info to check if symbol is valid
            try:
                info = ticker.info
                if not info or 'symbol' not in info:
                    self.logger.warning(f"Symbol {symbol} may be invalid or delisted")
                    return symbol, None
            except Exception as info_error:
                self.logger.warning(f"Could not get info for {symbol}: {info_error}")

            # Download historical data
            data = ticker.history(period=period, interval=interval)

            if data.empty:
                self.logger.warning(f"No historical data available for {symbol} - may be delisted or invalid")
                return symbol, None

            # Validate data quality
            if len(data) < 10:  # Minimum data points
                self.logger.warning(f"Insufficient data for {symbol} - only {len(data)} bars")
                return symbol, None

            # Check for recent data to avoid delisted stocks
            if not data.index.empty:
                last_date = data.index[-1].date() if hasattr(data.index[-1], 'date') else data.index[-1]
                days_old = (datetime.now().date() - last_date).days
                if days_old > 30:  # Data older than 30 days
                    self.logger.warning(f"Data for {symbol} is {days_old} days old - may be delisted")
                    return symbol, None

            data.reset_index(inplace=True)
            self.logger.info(f"Successfully downloaded {len(data)} bars for {symbol}")
            return symbol, data

        except Exception as e:
            error_msg = str(e).lower()
            if 'delisted' in error_msg or 'not found' in error_msg:
                self.logger.error(f"Symbol {symbol} appears to be delisted or not found: {e}")
            else:
                self.logger.error(f"Error downloading {symbol}: {e}")
            return symbol, None

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_symbol = {executor.submit(download_single, symbol): symbol
                            for symbol in symbols}

        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol, data = future.result()
            if data is not None:
                results[symbol] = data
                self.logger.info(f"Downloaded {len(data)} bars for {symbol}")

    return results

def _download_sequential(self, symbols: List[str], period: str,
                         interval: str) -> Dict[str, pd.DataFrame]:
    """Download data sequentially"""
    results = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)

            # Check if symbol is valid first
            try:
                info = ticker.info
                if not info or 'symbol' not in info:
                    self.logger.warning(f"Symbol {symbol} may be invalid or delisted")
                    continue
            except Exception as info_error:
                self.logger.warning(f"Could not get info for {symbol}: {info_error}")

            data = ticker.history(period=period, interval=interval)

            if data.empty:
                self.logger.warning(f"No data available for {symbol} - may be delisted")
                continue

            # Validate data quality
            if len(data) < 10:
                self.logger.warning(f"Insufficient data for {symbol}")
                continue

            # Check data recency
            if not data.index.empty:
                last_date = data.index[-1].date() if hasattr(data.index[-1], 'date') else data.index[-1]
                days_old = (datetime.now().date() - last_date).days
                if days_old > 30:
                    self.logger.warning(f"Data for {symbol} is {days_old} days old - may be delisted")
                    continue

            data.reset_index(inplace=True)
            results[symbol] = data
            self.logger.info(f"Downloaded {len(data)} bars for {symbol}")

        except Exception as e:
            error_msg = str(e).lower()
            if 'delisted' in error_msg or 'not found' in error_msg:
                self.logger.error(f"Symbol {symbol} appears to be delisted: {e}")
            else:
                self.logger.error(f"Error downloading {symbol}: {e}")
            continue

    return results

class HeikinAshiCalculator:
    """High-performance Heikin Ashi calculations using vectorized operations"""

    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Heikin Ashi values using vectorized operations
        Returns DataFrame with HA_Open, HA_High, HA_Low, HA_Close, HA_Color
        """
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
    """Generates trading signals based on Heikin Ashi patterns"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Generate trading signals from Heikin Ashi data"""
        signals = []

        if len(data) < 2:
            return signals

        # Vectorized signal detection
        prev_ha_color = data['HA_Color'].shift(1)
        curr_ha_color = data['HA_Color']

        # Long entry conditions (all on HA candles)
        long_cond1 = prev_ha_color == 'GREEN'  # Previous HA candle is green
        long_cond2 = data['HA_Close'] > data['HA_Close'].shift(1)  # Current HA close > prev HA close
        long_cond3 = np.abs(data['HA_Open'] - data['HA_Low']) < (data['HA_High'] - data['HA_Low']) * self.config.price_tolerance  # HA open = HA low

        long_entries = long_cond1 & long_cond2 & long_cond3

        # Short entry conditions (all on HA candles)
        short_cond1 = prev_ha_color == 'RED'  # Previous HA candle is red
        short_cond2 = np.abs(data['HA_Open'] - data['HA_High']) < (data['HA_High'] - data['HA_Low']) * self.config.price_tolerance  # HA open = HA high
        short_cond3 = data['HA_Close'] < data['HA_Close'].shift(1)  # Current HA close < prev HA close

        short_entries = short_cond1 & short_cond2 & short_cond3

        # Exit conditions
        green_to_red = (prev_ha_color == 'GREEN') & (curr_ha_color == 'RED')
        red_to_green = (prev_ha_color == 'RED') & (curr_ha_color == 'GREEN')

        # Generate signals
        for i in range(1, len(data)):
            timestamp = data.iloc[i]['Datetime'] if 'Datetime' in data.columns else data.index[i]

            # Ensure timestamp is timezone-aware in IST
            if hasattr(timestamp, 'tz') and timestamp.tz is None:
                timestamp = timestamp.tz_localize('UTC').tz_convert('Asia/Kolkata')
            elif not hasattr(timestamp, 'tz'):
                # If it's a regular datetime, convert to IST
                timestamp = pd.to_datetime(timestamp).tz_localize('UTC').tz_convert('Asia/Kolkata')

            if long_entries.iloc[i]:
                signals.append(TradeSignal(
                    timestamp=timestamp,
                    symbol='',  # Will be set by caller
                    signal_type='LONG',
                    entry_price=data.iloc[i]['Close'],  # Regular chart price
                    stop_loss=data.iloc[i]['Open'],     # Regular chart price
                    ha_close=data.iloc[i]['HA_Close'],
                    ha_open=data.iloc[i]['HA_Open'],
                    regular_close=data.iloc[i]['Close'],
                    regular_open=data.iloc[i]['Open']
                ))

            if short_entries.iloc[i]:
                signals.append(TradeSignal(
                    timestamp=timestamp,
                    symbol='',  # Will be set by caller
                    signal_type='SHORT',
                    entry_price=data.iloc[i]['Close'],  # Regular chart price
                    stop_loss=data.iloc[i]['Open'],     # Regular chart price
                    ha_close=data.iloc[i]['HA_Close'],
                    ha_open=data.iloc[i]['HA_Open'],
                    regular_close=data.iloc[i]['Close'],
                    regular_open=data.iloc[i]['Open']
                ))

            if green_to_red.iloc[i]:
                signals.append(TradeSignal(
                    timestamp=timestamp,
                    symbol='',
                    signal_type='EXIT_LONG',
                    entry_price=data.iloc[i]['Close'],
                    stop_loss=0,
                    ha_close=data.iloc[i]['HA_Close'],
                    ha_open=data.iloc[i]['HA_Open'],
                    regular_close=data.iloc[i]['Close'],
                    regular_open=data.iloc[i]['Open']
                ))

            if red_to_green.iloc[i]:
                signals.append(TradeSignal(
                    timestamp=timestamp,
                    symbol='',
                    signal_type='EXIT_SHORT',
                    entry_price=data.iloc[i]['Close'],
                    stop_loss=0,
                    ha_close=data.iloc[i]['HA_Close'],
                    ha_open=data.iloc[i]['HA_Open'],
                    regular_close=data.iloc[i]['Close'],
                    regular_open=data.iloc[i]['Open']
                ))

        return signals

class TradingEngine:
    """Main trading engine that orchestrates all components"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.data_manager = DataManager(config)
        self.signal_generator = SignalGenerator(config)
        self.trades: List[Trade] = []
        self.open_positions: Dict[str, Trade] = {}
        self.trade_id_counter = 0

        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)

    def setup_logging(self):
        """Setup comprehensive logging with IST timestamps"""
        # Create formatter with IST timezone
        ist_tz = pytz.timezone('Asia/Kolkata')

        class ISTFormatter(logging.Formatter):
            def converter(self, timestamp):
                dt = datetime.fromtimestamp(timestamp)
                return dt.replace(tzinfo=pytz.UTC).astimezone(ist_tz).timetuple()

        formatter = ISTFormatter(
            fmt='%(asctime)s IST - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Create handlers
        file_handler = logging.FileHandler(
            f'trading_engine_{datetime.now(ist_tz).strftime("%Y%m%d_%H%M%S")}_IST.log'
        )
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            handlers=[file_handler, console_handler]
        )

    def run_backtest(self, start_date: str = None, end_date: str = None) -> Dict:
        """Run historical backtest with multiple data sources"""
        start_time = datetime.now()
        self.logger.info("Starting backtest...")

        # Download data with API keys
        download_kwargs = {
            'alpha_vantage_key': self.config.alpha_vantage_key,
            'polygon_key': self.config.polygon_key,
            'skip_validation': True  # Skip yfinance validation to avoid rate limits
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
            # Calculate Heikin Ashi
            ha_data = HeikinAshiCalculator.calculate(data)

            # Generate signals
            signals = self.signal_generator.generate_signals(ha_data)

            # Set symbol for each signal
            for signal in signals:
                signal.symbol = symbol

            all_signals.extend(signals)
            self.logger.info(f"Generated {len(signals)} signals for {symbol}")

        # Sort signals by timestamp
        all_signals.sort(key=lambda x: x.timestamp)

        # Execute trades
        for signal in all_signals:
            self._process_signal(signal)

        # Calculate performance metrics
        performance = self._calculate_performance()

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        self.logger.info(f"Backtest completed in {execution_time:.2f} seconds")
        self.logger.info(f"Total trades: {len(self.trades)}")

        return {
            'trades': self.trades,
            'performance': performance,
            'execution_time': execution_time,
            'total_signals': len(all_signals),
            'data_sources_used': list(data_dict.keys())
        }

    def _process_signal(self, signal: TradeSignal):
        """Process individual trading signal"""
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

            self.logger.info(f"Opened {signal.signal_type} position for {symbol} at {signal.entry_price}")

        elif signal.signal_type in ['EXIT_LONG', 'EXIT_SHORT']:
            if symbol in self.open_positions:
                trade = self.open_positions[symbol]

                # Check if exit matches position type
                if (signal.signal_type == 'EXIT_LONG' and trade.side == 'LONG') or \
                        (signal.signal_type == 'EXIT_SHORT' and trade.side == 'SHORT'):

                    # Close position
                    trade.exit_time = signal.timestamp
                    trade.exit_price = signal.entry_price  # Using regular chart price
                    trade.exit_reason = 'HA_COLOR_CHANGE'
                    trade.status = 'CLOSED'

                    # Calculate P&L
                    if trade.side == 'LONG':
                        trade.pnl = trade.exit_price - trade.entry_price
                    else:  # SHORT
                        trade.pnl = trade.entry_price - trade.exit_price

                    del self.open_positions[symbol]

                    self.logger.info(f"Closed {trade.side} position for {symbol} at {trade.exit_price}, P&L: {trade.pnl:.2f}")

    def _calculate_performance(self) -> Dict:
        """Calculate performance metrics"""
        closed_trades = [t for t in self.trades if t.status == 'CLOSED']

        if not closed_trades:
            return {'total_trades': 0}

        pnls = [t.pnl for t in closed_trades]

        total_pnl = sum(pnls)
        win_trades = [p for p in pnls if p > 0]
        lose_trades = [p for p in pnls if p < 0]

        metrics = {
            'total_trades': len(closed_trades),
            'winning_trades': len(win_trades),
            'losing_trades': len(lose_trades),
            'win_rate': len(win_trades) / len(closed_trades) if closed_trades else 0,
            'total_pnl': total_pnl,
            'average_win': np.mean(win_trades) if win_trades else 0,
            'average_loss': np.mean(lose_trades) if lose_trades else 0,
            'profit_factor': abs(sum(win_trades) / sum(lose_trades)) if lose_trades and sum(lose_trades) != 0 else float('inf'),
            'max_win': max(pnls) if pnls else 0,
            'max_loss': min(pnls) if pnls else 0,
        }

        # Calculate drawdown
        cumulative_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        metrics['max_drawdown'] = np.max(drawdown) if len(drawdown) > 0 else 0

        return metrics

    def export_trades(self, filename: str = None):
        """Export trades to CSV with IST timestamps"""
        if not filename:
            ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
            filename = f"trades_{ist_now.strftime('%Y%m%d_%H%M%S')}_IST.csv"

        trades_data = []
        for trade in self.trades:
            # Ensure all timestamps are in IST
            entry_time_ist = trade.entry_time
            if hasattr(entry_time_ist, 'tz') and entry_time_ist.tz is None:
                entry_time_ist = entry_time_ist.tz_localize('UTC').tz_convert('Asia/Kolkata')

            exit_time_ist = trade.exit_time
            if exit_time_ist and hasattr(exit_time_ist, 'tz') and exit_time_ist.tz is None:
                exit_time_ist = exit_time_ist.tz_localize('UTC').tz_convert('Asia/Kolkata')

            trades_data.append({
                'id': trade.id,
                'symbol': trade.symbol,
                'side': trade.side,
                'entry_time_ist': entry_time_ist,
                'entry_price': trade.entry_price,
                'stop_loss': trade.stop_loss,
                'exit_time_ist': exit_time_ist,
                'exit_price': trade.exit_price,
                'exit_reason': trade.exit_reason,
                'pnl': trade.pnl,
                'status': trade.status,
                'timezone': 'Asia/Kolkata (+05:30)'
            })

        df = pd.DataFrame(trades_data)
        df.to_csv(filename, index=False)
        self.logger.info(f"Trades exported to {filename} with IST timestamps")

def main():
    """Main execution function with multiple data source options"""
    # Initialize configuration
    config = TradingConfig()

    # Set your API keys here (get free keys from the respective websites)
    config.alpha_vantage_key = "demo"  # Replace with your free Alpha Vantage API key
    config.polygon_key = "demo"        # Replace with your free Polygon API key

    # Use a mix of active stocks
    config.symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    config.timeframe = '5m'
    config.concurrent_downloads = False  # Set to False to avoid overwhelming APIs

    print("=== MULTI-SOURCE DATA DOWNLOADER WITH IST TIMEZONE ===")
    print(f"Timezone: {config.timezone} (IST +05:30)")
    print(f"Current IST time: {datetime.now(config.ist_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print()
    print("Data sources available (in priority order):")
    print("1. Alpha Vantage (free: 5 calls/min, 500 calls/day)")
    print("2. Polygon.io (free: 5 calls/min)")
    print("3. Yahoo Direct API (rate limited)")
    print("4. yfinance (fallback, prone to 429 errors)")
    print()
    print("All timestamps will be converted to IST timezone")
    print("To avoid 429 errors:")
    print("- Get free API keys from Alpha Vantage and/or Polygon")
    print("- Set config.alpha_vantage_key and config.polygon_key")
    print("- Reduce concurrent downloads or add delays")
    print()

    # Initialize trading engine
    engine = TradingEngine(config)

    try:
        # Run backtest
        results = engine.run_backtest()

        # Print performance summary
        performance = results['performance']
        print("\n=== BACKTEST RESULTS ===")
        print(f"Data Sources Used: {results.get('data_sources_used', [])}")
        print(f"Execution Time: {results['execution_time']:.2f} seconds")
        print(f"Total Signals: {results['total_signals']}")
        print(f"Total Trades: {performance.get('total_trades', 0)}")

        if performance.get('total_trades', 0) > 0:
            print(f"Win Rate: {performance.get('win_rate', 0):.2%}")
            print(f"Total P&L: ${performance.get('total_pnl', 0):.2f}")
            print(f"Max Drawdown: ${performance.get('max_drawdown', 0):.2f}")
            print(f"Profit Factor: {performance.get('profit_factor', 0):.2f}")
            print(f"Average Win: ${performance.get('average_win', 0):.2f}")
            print(f"Average Loss: ${performance.get('average_loss', 0):.2f}")
        else:
            print("No completed trades found.")

        # Export trades if any exist
        if len(engine.trades) > 0:
            engine.export_trades()
            print(f"\nTrade history exported to CSV with IST timestamps")

        print("\n=== TIMEZONE INFORMATION ===")
        print("All data timestamps have been converted to IST (Asia/Kolkata)")
        print("Log files include IST timestamps")
        print("Exported CSV files contain IST timezone information")

        print("\n=== API KEY SETUP INSTRUCTIONS ===")
        print("To avoid rate limiting issues:")
        print("1. Alpha Vantage: https://www.alphavantage.co/support/#api-key")
        print("2. Polygon.io: https://polygon.io/ (free tier available)")
        print("3. Set the keys in TradingConfig before running")

        return engine

    except Exception as e:
        print(f"Error during backtest: {e}")
        import traceback
        traceback.print_exc()
        return None)}")

        if performance.get('total_trades', 0) > 0:
            print(f"Win Rate: {performance.get('win_rate', 0):.2%}")
        print(f"Total P&L: ${performance.get('total_pnl', 0):.2f}")
        print(f"Max Drawdown: ${performance.get('max_drawdown', 0):.2f}")
        print(f"Profit Factor: {performance.get('profit_factor', 0):.2f}")
        print(f"Average Win: ${performance.get('average_win', 0):.2f}")
        print(f"Average Loss: ${performance.get('average_loss', 0):.2f}")
        else:
        print("No completed trades found.")

        # Export trades if any exist
        if len(engine.trades) > 0:
            engine.export_trades()
        print(f"\nTrade history exported to CSV")

        print("\n=== API KEY SETUP INSTRUCTIONS ===")
        print("To avoid rate limiting issues:")
        print("1. Alpha Vantage: https://www.alphavantage.co/support/#api-key")
        print("2. Polygon.io: https://polygon.io/ (free tier available)")
        print("3. Set the keys in TradingConfig before running")

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
    print("Trading engine completed successfully!")