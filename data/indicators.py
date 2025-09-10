"""
Technical indicators calculations
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging


class HeikinAshiCalculator:
    """High-performance Heikin Ashi calculations using vectorized operations"""

    @staticmethod
    def calculate(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate Heikin Ashi values using vectorized operations"""
        logger = logging.getLogger(__name__)

        if df is None or df.empty:
            logger.warning("Empty or None DataFrame provided to HeikinAshi calculator")
            return df

        # Validate required columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns for Heikin Ashi: {missing_columns}")
            return df

        try:
            data = df.copy()
            n = len(data)

            if n == 0:
                return data

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

            logger.debug(f"Calculated Heikin Ashi for {n} bars")
            return data

        except Exception as e:
            logger.error(f"Error calculating Heikin Ashi: {e}")
            return df

    @staticmethod
    def validate_ha_data(df: pd.DataFrame) -> bool:
        """Validate that DataFrame contains proper Heikin Ashi data"""
        required_ha_columns = ['HA_Open', 'HA_High', 'HA_Low', 'HA_Close', 'HA_Color']
        return all(col in df.columns for col in required_ha_columns)


class MovingAverages:
    """Moving average calculations"""

    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()

    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period).mean()

    @staticmethod
    def wma(data: pd.Series, period: int) -> pd.Series:
        """Weighted Moving Average"""
        weights = np.arange(1, period + 1)
        return data.rolling(period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )


class TechnicalIndicators:
    """Collection of technical indicators"""

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Bollinger Bands"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()

        return pd.DataFrame({
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev)
        })

    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                   k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()

        return pd.DataFrame({
            'k_percent': k_percent,
            'd_percent': d_percent
        })

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
        """
        Average Directional Index (ADX) and Directional Indicators (DI+ and DI-)

        ADX measures trend strength (0-100):
        - 0-25: Weak or no trend
        - 25-50: Strong trend
        - 50-75: Very strong trend
        - 75-100: Extremely strong trend

        DI+ and DI- show trend direction:
        - DI+ > DI-: Uptrend
        - DI- > DI+: Downtrend

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Period for calculation (default 14)

        Returns:
            DataFrame with columns: adx, di_plus, di_minus
        """
        # Calculate True Range (TR)
        high_low = high - low
        high_close_prev = abs(high - close.shift(1))
        low_close_prev = abs(low - close.shift(1))
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

        # Calculate Directional Movement (DM)
        high_diff = high - high.shift(1)
        low_diff = low.shift(1) - low

        # Positive Directional Movement (+DM)
        plus_dm = pd.Series(index=high.index, dtype=float)
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)

        # Negative Directional Movement (-DM)
        minus_dm = pd.Series(index=low.index, dtype=float)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        # Smooth TR, +DM, and -DM using Wilder's smoothing (exponential moving average)
        alpha = 1.0 / period

        # Initialize smoothed values
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()

        # Calculate Directional Indicators (DI+ and DI-)
        di_plus = 100 * (plus_dm_smooth / atr)
        di_minus = 100 * (minus_dm_smooth / atr)

        # Calculate Directional Index (DX)
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)

        # Calculate ADX (smoothed DX)
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        return pd.DataFrame({
            'adx': adx,
            'di_plus': di_plus,
            'di_minus': di_minus
        })

    @staticmethod
    def adx_trend_strength(adx_value: float) -> str:
        """
        Classify ADX trend strength

        Args:
            adx_value: ADX value (0-100)

        Returns:
            String description of trend strength
        """
        if adx_value < 25:
            return "Weak/No Trend"
        elif adx_value < 50:
            return "Strong Trend"
        elif adx_value < 75:
            return "Very Strong Trend"
        else:
            return "Extremely Strong Trend"

    @staticmethod
    def adx_signals(adx_df: pd.DataFrame, adx_threshold: float = 25.0) -> pd.Series:
        """
        Generate basic ADX trading signals

        Args:
            adx_df: DataFrame with ADX data (from adx() method)
            adx_threshold: Minimum ADX value to consider trend strong enough

        Returns:
            Series with signals: 'LONG', 'SHORT', 'NO_TREND', or None
        """
        signals = pd.Series(index=adx_df.index, dtype=str)

        # Strong trend conditions
        strong_trend = adx_df['adx'] >= adx_threshold

        # Direction conditions
        bullish = adx_df['di_plus'] > adx_df['di_minus']
        bearish = adx_df['di_minus'] > adx_df['di_plus']

        # Generate signals
        signals.loc[strong_trend & bullish] = 'LONG'
        signals.loc[strong_trend & bearish] = 'SHORT'
        signals.loc[~strong_trend] = 'NO_TREND'

        return signals


class PatternRecognition:
    """Price pattern recognition utilities"""

    @staticmethod
    def is_doji(open_price: float, close_price: float, high: float, low: float,
                tolerance: float = 0.001) -> bool:
        """Check if candle is a doji"""
        body_size = abs(close_price - open_price)
        total_range = high - low
        return body_size <= (total_range * tolerance)