"""
Trading signal generation logic
"""

import numpy as np
import pandas as pd
from typing import List
import logging

from config import TradingConfig
from core.models import TradeSignal
from utils.time_manager import TimeManager
from utils.logger import LoggerMixin


class SignalGenerator(LoggerMixin):
    """Generates trading signals based on Heikin Ashi patterns with EOD and Stop Loss logic"""

    def __init__(self, config: TradingConfig):
        super().__init__()
        self.config = config
        self.time_manager = TimeManager(config)

    def generate_signals(self, data: pd.DataFrame, symbol: str) -> List[TradeSignal]:
        """Generate trading signals from Heikin Ashi data"""
        signals = []

        if data is None or len(data) < 2:
            self.log_warning(f"Insufficient data for signal generation: {symbol}")
            return signals

        # Validate required columns
        if not self._validate_data_columns(data, symbol):
            return signals

        try:
            # Generate different types of signals
            entry_signals = self._generate_entry_signals(data, symbol)
            exit_signals = self._generate_exit_signals(data, symbol)
            eod_signals = self._generate_eod_signals(data, symbol)
            stop_loss_signals = self._generate_stop_loss_signals(data, symbol)

            # Combine all signals
            signals.extend(entry_signals)
            signals.extend(exit_signals)
            signals.extend(eod_signals)
            signals.extend(stop_loss_signals)

            self.log_info(f"Generated {len(signals)} signals for {symbol}")

        except Exception as e:
            self.log_error(f"Error generating signals for {symbol}: {e}", exc_info=True)

        return signals

    def _validate_data_columns(self, data: pd.DataFrame, symbol: str) -> bool:
        """Validate that data has required columns"""
        required_columns = ['Open', 'High', 'Low', 'Close', 'HA_Open', 'HA_High', 'HA_Low', 'HA_Close', 'HA_Color']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            self.log_error(f"Missing required columns for {symbol}: {missing_columns}")
            return False
        return True

    def _generate_entry_signals(self, data: pd.DataFrame, symbol: str) -> List[TradeSignal]:
        """Generate entry signals (LONG/SHORT)"""
        signals = []

        # Vectorized signal detection
        prev_ha_color = data['HA_Color'].shift(1)

        # Long entry conditions
        long_cond1 = prev_ha_color == 'GREEN'
        long_cond2 = data['HA_Close'] > data['HA_Close'].shift(1)
        long_cond3 = np.abs(data['HA_Open'] - data['HA_Low']) < (data['HA_High'] - data['HA_Low']) * self.config.price_tolerance
        long_entries = long_cond1 & long_cond2 & long_cond3

        # Short entry conditions
        short_cond1 = prev_ha_color == 'RED'
        short_cond2 = np.abs(data['HA_Open'] - data['HA_High']) < (data['HA_High'] - data['HA_Low']) * self.config.price_tolerance
        short_cond3 = data['HA_Close'] < data['HA_Close'].shift(1)
        short_entries = short_cond1 & short_cond2 & short_cond3

        # Generate entry signals
        for i in range(1, len(data)):
            timestamp = self._get_timestamp(data, i)

            if not self._is_market_open_for_entry(timestamp, symbol):
                continue

            if long_entries.iloc[i]:
                signal = self._create_signal(data, i, symbol, 'LONG', timestamp)
                if signal:
                    signals.append(signal)

            if short_entries.iloc[i]:
                signal = self._create_signal(data, i, symbol, 'SHORT', timestamp)
                if signal:
                    signals.append(signal)

        return signals

    def _generate_exit_signals(self, data: pd.DataFrame, symbol: str) -> List[TradeSignal]:
        """Generate exit signals based on HA color changes"""
        signals = []

        prev_ha_color = data['HA_Color'].shift(1)
        curr_ha_color = data['HA_Color']

        # Exit conditions
        green_to_red = (prev_ha_color == 'GREEN') & (curr_ha_color == 'RED')
        red_to_green = (prev_ha_color == 'RED') & (curr_ha_color == 'GREEN')

        for i in range(1, len(data)):
            timestamp = self._get_timestamp(data, i)

            if green_to_red.iloc[i]:
                signal = self._create_signal(data, i, symbol, 'EXIT_LONG', timestamp)
                if signal:
                    signals.append(signal)

            if red_to_green.iloc[i]:
                signal = self._create_signal(data, i, symbol, 'EXIT_SHORT', timestamp)
                if signal:
                    signals.append(signal)

        return signals

    def _generate_eod_signals(self, data: pd.DataFrame, symbol: str) -> List[TradeSignal]:
        """Generate end-of-day exit signals"""
        signals = []

        for i in range(1, len(data)):
            timestamp = self._get_timestamp(data, i)

            if self.time_manager.should_exit_eod(timestamp, symbol):
                signal = self._create_signal(data, i, symbol, 'EOD_EXIT', timestamp)
                if signal:
                    signals.append(signal)

        return signals

    def _generate_stop_loss_signals(self, data: pd.DataFrame, symbol: str) -> List[TradeSignal]:
        """Generate stop loss check signals"""
        signals = []

        for i in range(1, len(data)):
            timestamp = self._get_timestamp(data, i)

            if self.time_manager.is_market_open(timestamp, symbol):
                signal = self._create_signal(data, i, symbol, 'STOP_LOSS', timestamp)
                if signal:
                    signals.append(signal)

        return signals

    def _get_timestamp(self, data: pd.DataFrame, index: int) -> pd.Timestamp:
        """Get timestamp for given index"""
        if 'Datetime' in data.columns:
            timestamp = data.iloc[index]['Datetime']
        else:
            timestamp = data.index[index]

        # Ensure timestamp is timezone-aware
        try:
            if hasattr(timestamp, 'tz_localize') and timestamp.tz is None:
                timestamp = timestamp.tz_localize(self.time_manager.utc_tz).astimezone(self.time_manager.ist_tz)
            elif hasattr(timestamp, 'astimezone'):
                timestamp = timestamp.astimezone(self.time_manager.ist_tz)
        except Exception as e:
            self.log_warning(f"Timestamp conversion warning: {e}")

        return timestamp

    def _is_market_open_for_entry(self, timestamp, symbol: str) -> bool:
        """Check if market is open and not near EOD for entries"""
        return (self.time_manager.is_market_open(timestamp, symbol) and
                not self.time_manager.should_exit_eod(timestamp, symbol))

    def _create_signal(self, data: pd.DataFrame, index: int, symbol: str,
                       signal_type: str, timestamp) -> TradeSignal:
        """Create a trading signal from data"""
        try:
            row = data.iloc[index]

            # Calculate stop loss based on signal type
            if signal_type == 'LONG':
                stop_loss = min(float(row['Open']), float(row['HA_Open']))
            elif signal_type == 'SHORT':
                stop_loss = max(float(row['Open']), float(row['HA_Open']))
            else:
                stop_loss = 0.0000

            signal = TradeSignal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=signal_type,
                entry_price=round(float(row['Close']), 4),
                stop_loss=round(stop_loss, 4),
                ha_close=round(float(row['HA_Close']), 4),
                ha_open=round(float(row['HA_Open']), 4),
                regular_close=round(float(row['Close']), 4),
                regular_open=round(float(row['Open']), 4)
            )

            return signal

        except Exception as e:
            self.log_error(f"Error creating signal for {symbol}: {e}")
            return None


class SignalFilter:
    """Filters and validates trading signals"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def filter_signals(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """Filter signals based on various criteria"""
        filtered_signals = []

        for signal in signals:
            if self._validate_signal(signal):
                filtered_signals.append(signal)

        self.logger.info(f"Filtered {len(signals)} signals to {len(filtered_signals)}")
        return filtered_signals

    def _validate_signal(self, signal: TradeSignal) -> bool:
        """Validate individual signal"""
        # Basic validation
        if signal.entry_price <= 0:
            return False

        # Stop loss validation for entry signals
        if signal.is_entry_signal():
            if signal.signal_type == 'LONG' and signal.stop_loss >= signal.entry_price:
                return False
            if signal.signal_type == 'SHORT' and signal.stop_loss <= signal.entry_price:
                return False

        return True


class SignalAnalyzer:
    """Analyzes signal patterns and statistics"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_signals(self, signals: List[TradeSignal]) -> dict:
        """Analyze signal distribution and patterns"""
        if not signals:
            return {}

        analysis = {
            'total_signals': len(signals),
            'by_type': {},
            'by_symbol': {},
            'time_distribution': {}
        }

        # Count by signal type
        for signal in signals:
            signal_type = signal.signal_type
            symbol = signal.symbol

            analysis['by_type'][signal_type] = analysis['by_type'].get(signal_type, 0) + 1
            analysis['by_symbol'][symbol] = analysis['by_symbol'].get(symbol, 0) + 1

        # Calculate ratios
        entry_signals = analysis['by_type'].get('LONG', 0) + analysis['by_type'].get('SHORT', 0)
        exit_signals = (analysis['by_type'].get('EXIT_LONG', 0) +
                        analysis['by_type'].get('EXIT_SHORT', 0) +
                        analysis['by_type'].get('EOD_EXIT', 0))

        analysis['entry_exit_ratio'] = exit_signals / entry_signals if entry_signals > 0 else 0

        return analysis