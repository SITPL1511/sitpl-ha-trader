"""
Data source implementations for various financial data providers
"""

import pandas as pd
import yfinance as yf
import requests
import time
import random
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

from utils.time_manager import TimeManager
from config import TradingConfig


class DataSourceBase:
    """Base class for data sources"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.time_manager = TimeManager(config)
        self.logger = logging.getLogger(self.__class__.__name__)

    def download(self, symbol: str, **kwargs) -> Optional[pd.DataFrame]:
        """Download data - to be implemented by subclasses"""
        raise NotImplementedError

    def convert_timestamps_to_ist(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert all timestamps in dataframe to IST"""
        if df is None or df.empty:
            return df

        try:
            if 'Datetime' in df.columns:
                df['Datetime'] = df['Datetime'].apply(self.time_manager.convert_to_ist)
            elif df.index.name == 'Datetime' or isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.DatetimeIndex([self.time_manager.convert_to_ist(ts) for ts in df.index])
        except Exception as e:
            self.logger.error(f"Error converting timestamps to IST: {e}")

        return df


class AlphaVantageSource(DataSourceBase):
    """Alpha Vantage data source"""

    def __init__(self, config: TradingConfig):
        super().__init__(config)
        self.base_url = "https://www.alphavantage.co/query"
        self.session = requests.Session()

    def download(self, symbol: str, api_key: str = "demo",
                 interval: str = "5min", outputsize: str = "compact") -> Optional[pd.DataFrame]:
        """Download data from Alpha Vantage API"""
        try:
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

            response = self.session.get(self.base_url, params=params)
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

            if not df_data:
                self.logger.error(f"No valid data found for {symbol}")
                return None

            df = pd.DataFrame(df_data)
            df = df.sort_values('Datetime').reset_index(drop=True)
            df = self.convert_timestamps_to_ist(df)

            self.logger.info(f"Downloaded {len(df)} bars for {symbol} from Alpha Vantage")
            return df

        except Exception as e:
            self.logger.error(f"Alpha Vantage error for {symbol}: {e}")
            return None


class PolygonSource(DataSourceBase):
    """Polygon.io data source"""

    def __init__(self, config: TradingConfig):
        super().__init__(config)
        self.session = requests.Session()

    def download(self, symbol: str, api_key: str = "demo",
                 multiplier: int = 5, timespan: str = "minute",
                 from_date: str = None, to_date: str = None) -> Optional[pd.DataFrame]:
        """Download data from Polygon.io API"""
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

            if not df_data:
                self.logger.error(f"No valid data found for {symbol}")
                return None

            df = pd.DataFrame(df_data)
            df = df.sort_values('Datetime').reset_index(drop=True)
            df = self.convert_timestamps_to_ist(df)

            self.logger.info(f"Downloaded {len(df)} bars for {symbol} from Polygon")
            return df

        except Exception as e:
            self.logger.error(f"Polygon error for {symbol}: {e}")
            return None


class YahooDirectSource(DataSourceBase):
    """Direct Yahoo Finance API access"""

    def __init__(self, config: TradingConfig):
        super().__init__(config)
        self.session = requests.Session()
        self._setup_session()

    def _setup_session(self):
        """Setup requests session with headers"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        self.session.headers.update(headers)

    def download(self, symbol: str, period1: int = None,
                 period2: int = None, interval: str = "5m") -> Optional[pd.DataFrame]:
        """Direct Yahoo Finance API access"""
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
            df = self.convert_timestamps_to_ist(df)

            self.logger.info(f"Downloaded {len(df)} bars for {symbol} from Yahoo Direct")
            return df

        except Exception as e:
            self.logger.error(f"Yahoo Direct error for {symbol}: {e}")
            return None


class YFinanceSource(DataSourceBase):
    """yfinance library wrapper"""

    def download(self, symbol: str, max_retries: int = 3, **kwargs) -> Optional[pd.DataFrame]:
        """yfinance with exponential backoff retry logic"""
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

                    # Ensure Datetime column exists
                    if 'Datetime' not in data.columns and isinstance(data.index, pd.DatetimeIndex):
                        data['Datetime'] = data.index
                        data.reset_index(drop=True, inplace=True)

                    data = self.convert_timestamps_to_ist(data)
                    self.logger.info(f"Downloaded {len(data)} bars for {symbol} from yfinance")
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


class DataSourceFactory:
    """Factory for creating data source instances"""

    @staticmethod
    def create_source(source_type: str, config: TradingConfig) -> DataSourceBase:
        """Create data source instance based on type"""
        sources = {
            'alpha_vantage': AlphaVantageSource,
            'polygon': PolygonSource,
            'yahoo_direct': YahooDirectSource,
            'yfinance': YFinanceSource
        }

        if source_type not in sources:
            raise ValueError(f"Unknown data source type: {source_type}")

        return sources[source_type](config)