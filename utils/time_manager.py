"""
Time and timezone management utilities
"""

import pytz
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import logging

from config import TradingConfig


class TimeManager:
    """Handles timezone conversions and market hours logic"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.ist_tz = pytz.timezone('Asia/Kolkata')
        self.utc_tz = pytz.UTC
        self.logger = logging.getLogger(__name__)

    def convert_to_ist(self, timestamp) -> Optional[datetime]:
        """Convert any timestamp to IST timezone"""
        if timestamp is None:
            return None

        try:
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

        except Exception as e:
            self.logger.error(f"Error converting timestamp to IST: {e}")
            return None

    def is_market_open(self, timestamp: datetime, symbol: str) -> bool:
        """Check if market is open for given symbol at given time"""
        try:
            market = self.config.detect_market(symbol)

            if market not in self.config.market_hours:
                return True  # Default to always open for unknown markets

            market_config = self.config.market_hours[market]
            ist_time = self.convert_to_ist(timestamp)

            if ist_time is None:
                return False

            current_time = ist_time.time()

            # Handle markets that close next day (like NYSE)
            if market_config['close'] < market_config['open']:
                # Market spans midnight
                return (current_time >= market_config['open'] or
                        current_time <= market_config['close'])
            else:
                # Regular market hours
                return (market_config['open'] <= current_time <=
                        market_config['close'])

        except Exception as e:
            self.logger.error(f"Error checking market hours for {symbol}: {e}")
            return True  # Default to open on error

    def should_exit_eod(self, timestamp: datetime, symbol: str) -> bool:
        """Check if position should be closed due to EOD"""
        if not self.config.enable_eod_closure:
            return False

        try:
            market = self.config.detect_market(symbol)

            if market not in self.config.market_hours:
                return False

            market_config = self.config.market_hours[market]
            ist_time = self.convert_to_ist(timestamp)

            if ist_time is None:
                return False

            current_time = ist_time.time()

            # Check if current time is past EOD exit time
            eod_exit = market_config['eod_exit_time']

            if market_config['close'] < market_config['open']:
                # Market spans midnight - need special handling
                if eod_exit < market_config['open']:
                    # EOD exit is next day
                    return (current_time >= eod_exit and
                            current_time <= market_config['close'])
                else:
                    # EOD exit is same day
                    return current_time >= eod_exit
            else:
                # Regular market hours
                return current_time >= eod_exit

        except Exception as e:
            self.logger.error(f"Error checking EOD for {symbol}: {e}")
            return False

    def get_next_eod_time(self, timestamp: datetime, symbol: str) -> datetime:
        """Get the next EOD exit time for a symbol"""
        try:
            market = self.config.detect_market(symbol)

            if market not in self.config.market_hours:
                # Default: assume next day 3:15 PM IST
                ist_time = self.convert_to_ist(timestamp)
                if ist_time is None:
                    return timestamp + timedelta(days=1)

                next_eod = ist_time.replace(hour=15, minute=15, second=0, microsecond=0)
                if next_eod <= ist_time:
                    next_eod += timedelta(days=1)
                return next_eod

            market_config = self.config.market_hours[market]
            ist_time = self.convert_to_ist(timestamp)

            if ist_time is None:
                return timestamp + timedelta(days=1)

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

        except Exception as e:
            self.logger.error(f"Error calculating next EOD time for {symbol}: {e}")
            return timestamp + timedelta(days=1)

    def format_ist_timestamp(self, timestamp: datetime) -> str:
        """Format timestamp as IST string"""
        try:
            ist_time = self.convert_to_ist(timestamp)
            if ist_time:
                return ist_time.strftime('%Y-%m-%d %H:%M:%S IST')
            return "Invalid timestamp"
        except Exception as e:
            self.logger.error(f"Error formatting timestamp: {e}")
            return "Error formatting timestamp"

    def get_market_info(self, symbol: str) -> dict:
        """Get market information for a symbol"""
        market = self.config.detect_market(symbol)
        market_config = self.config.market_hours.get(market, {})

        return {
            'market': market,
            'open_time': market_config.get('open'),
            'close_time': market_config.get('close'),
            'eod_exit_time': market_config.get('eod_exit_time'),
            'spans_midnight': (market_config.get('close', None) and
                               market_config.get('open', None) and
                               market_config['close'] < market_config['open'])
        }