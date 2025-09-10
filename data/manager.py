"""
Data management and download coordination
"""

import pandas as pd
import yfinance as yf
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import concurrent.futures
import logging

from config import TradingConfig
from data.sources import DataSourceFactory
from utils.logger import LoggerMixin


class DataValidator:
    """Data validation utilities"""

    @staticmethod
    def validate_symbols(symbols: List[str]) -> List[str]:
        """Validate symbols and filter out delisted/invalid ones"""
        logger = logging.getLogger(__name__)
        valid_symbols = []

        logger.info("Validating symbols...")

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="5d", interval="1d")

                if not data.empty and len(data) >= 1:
                    # Handle both timezone-aware and timezone-naive dates
                    last_date = data.index[-1]
                    if hasattr(last_date, 'date'):
                        last_date = last_date.date()
                    else:
                        last_date = pd.to_datetime(last_date).date()

                    days_old = (datetime.now().date() - last_date).days

                    if days_old <= 7:
                        valid_symbols.append(symbol)
                        logger.info(f"✓ {symbol} is valid (last data: {days_old} days ago)")
                    else:
                        logger.warning(f"✗ {symbol} may be delisted (last data: {days_old} days ago)")
                else:
                    logger.warning(f"✗ {symbol} has no recent data - likely delisted")

            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['delisted', 'not found', 'invalid', '404']):
                    logger.error(f"✗ {symbol} is delisted or invalid: {e}")
                else:
                    logger.error(f"✗ Error validating {symbol}: {e}")

        logger.info(f"Validation complete: {len(valid_symbols)}/{len(symbols)} symbols are valid")
        return valid_symbols

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, symbol: str) -> bool:
        """Validate downloaded dataframe"""
        if df is None or df.empty:
            return False

        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger = logging.getLogger(__name__)
            logger.error(f"Missing columns for {symbol}: {missing_columns}")
            return False

        return True


class DataManager(LoggerMixin):
    """Manages data downloading from multiple sources with fallback options"""

    def __init__(self, config: TradingConfig):
        super().__init__()
        self.config = config
        self.data_sources = {}
        self._initialize_sources()

    def _initialize_sources(self):
        """Initialize all data sources"""
        for source_type in self.config.data_sources:
            try:
                self.data_sources[source_type] = DataSourceFactory.create_source(source_type, self.config)
                self.log_debug(f"Initialized {source_type} data source")
            except Exception as e:
                self.log_error(f"Failed to initialize {source_type} source: {e}")

    def download_data_with_fallback(self, symbol: str, **kwargs) -> Optional[pd.DataFrame]:
        """Try multiple data sources with fallback logic"""

        # Method 1: Alpha Vantage (if API key provided)
        if self._should_use_alpha_vantage(kwargs):
            data = self._try_alpha_vantage(symbol, kwargs)
            if data is not None and not data.empty:
                return data
            time.sleep(1)

        # Method 2: Polygon (if API key provided)
        if self._should_use_polygon(kwargs):
            data = self._try_polygon(symbol, kwargs)
            if data is not None and not data.empty:
                return data
            time.sleep(1)

        # Method 3: Yahoo Direct API
        data = self._try_yahoo_direct(symbol, kwargs)
        if data is not None and not data.empty:
            return data

        # Method 4: yfinance as last resort
        data = self._try_yfinance(symbol, kwargs)
        if data is not None and not data.empty:
            return data

        self.log_error(f"All data sources failed for {symbol}")
        return None

    def _should_use_alpha_vantage(self, kwargs: dict) -> bool:
        """Check if Alpha Vantage should be used"""
        api_key = kwargs.get('alpha_vantage_key', self.config.alpha_vantage_key)
        return (api_key and api_key != "demo" and
                'alpha_vantage' in self.data_sources)

    def _should_use_polygon(self, kwargs: dict) -> bool:
        """Check if Polygon should be used"""
        api_key = kwargs.get('polygon_key', self.config.polygon_key)
        return (api_key and api_key != "demo" and
                'polygon' in self.data_sources)

    def _try_alpha_vantage(self, symbol: str, kwargs: dict) -> Optional[pd.DataFrame]:
        """Try Alpha Vantage data source"""
        self.log_info(f"Trying Alpha Vantage for {symbol}")
        source = self.data_sources.get('alpha_vantage')
        if source:
            api_key = kwargs.get('alpha_vantage_key', self.config.alpha_vantage_key)
            return source.download(symbol, api_key, kwargs.get('interval', '5m'))
        return None

    def _try_polygon(self, symbol: str, kwargs: dict) -> Optional[pd.DataFrame]:
        """Try Polygon data source"""
        self.log_info(f"Trying Polygon for {symbol}")
        source = self.data_sources.get('polygon')
        if source:
            api_key = kwargs.get('polygon_key', self.config.polygon_key)
            return source.download(symbol, api_key, timespan=kwargs.get('timespan', 'minute'))
        return None

    def _try_yahoo_direct(self, symbol: str, kwargs: dict) -> Optional[pd.DataFrame]:
        """Try Yahoo Direct data source"""
        self.log_info(f"Trying Yahoo Direct API for {symbol}")
        source = self.data_sources.get('yahoo_direct')
        if source:
            return source.download(symbol, interval=kwargs.get('interval', '5m'))
        return None

    def _try_yfinance(self, symbol: str, kwargs: dict) -> Optional[pd.DataFrame]:
        """Try yfinance data source"""
        self.log_info(f"Trying yfinance as fallback for {symbol}")
        source = self.data_sources.get('yfinance')
        if source:
            return source.download(symbol, **kwargs)
        return None

    def download_data(self, symbols: List[str], period: str = "1y",
                      interval: str = "5m", **kwargs) -> Dict[str, pd.DataFrame]:
        """Download historical data for multiple symbols"""

        # Validate symbols unless skipped
        if not kwargs.get('skip_validation', False):
            valid_symbols = DataValidator.validate_symbols(symbols)
        else:
            valid_symbols = symbols

        if not valid_symbols:
            self.log_error("No valid symbols found!")
            return {}

        # Choose download method
        if self.config.concurrent_downloads:
            return self._download_concurrent(valid_symbols, period, interval, **kwargs)
        else:
            return self._download_sequential(valid_symbols, period, interval, **kwargs)

    def _download_concurrent(self, symbols: List[str], period: str,
                             interval: str, **kwargs) -> Dict[str, pd.DataFrame]:
        """Download using multiple sources concurrently"""
        def download_single(symbol):
            return symbol, self.download_data_with_fallback(
                symbol, period=period, interval=interval, **kwargs
            )

        results = {}
        max_workers = min(3, len(symbols))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(download_single, symbol): symbol
                for symbol in symbols
            }

            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol, data = future.result()
                if DataValidator.validate_dataframe(data, symbol):
                    results[symbol] = data
                time.sleep(0.5)

        return results

    def _download_sequential(self, symbols: List[str], period: str,
                             interval: str, **kwargs) -> Dict[str, pd.DataFrame]:
        """Download using fallback sources sequentially"""
        results = {}

        for i, symbol in enumerate(symbols):
            self.log_info(f"Processing {symbol} ({i+1}/{len(symbols)})")

            data = self.download_data_with_fallback(
                symbol, period=period, interval=interval, **kwargs
            )

            if DataValidator.validate_dataframe(data, symbol):
                results[symbol] = data

            if i < len(symbols) - 1:
                delay = random.uniform(1, 3)
                time.sleep(delay)

        return results

    def get_data_summary(self, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """Get summary of downloaded data"""
        summary = {
            'total_symbols': len(data_dict),
            'symbols': list(data_dict.keys()),
            'data_points': {symbol: len(df) for symbol, df in data_dict.items()},
            'date_ranges': {}
        }

        for symbol, df in data_dict.items():
            if not df.empty:
                if 'Datetime' in df.columns:
                    start_date = df['Datetime'].min()
                    end_date = df['Datetime'].max()
                elif isinstance(df.index, pd.DatetimeIndex):
                    start_date = df.index.min()
                    end_date = df.index.max()
                else:
                    start_date = end_date = "Unknown"

                summary['date_ranges'][symbol] = {
                    'start': start_date,
                    'end': end_date
                }

        return summary