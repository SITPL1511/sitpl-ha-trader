"""
Trading Engine Configuration
Contains all configuration settings and constants
"""

from datetime import time as dt_time
import pytz
from typing import List, Dict


class TradingConfig:
    """Trading engine configuration class"""

    def __init__(self):
        # Trading parameters
        self.symbols: List[str] = ['AAPL', 'MSFT', 'GOOGL']
        self.timeframe: str = '5m'  # 1m, 5m, 15m, 30m, 1h, 1d
        self.position_size: float = 1000.0  # USD per trade
        self.max_positions: int = 5
        self.risk_per_trade: float = 0.03  # 3% risk per trade
        self.data_lookback_days: int = 252
        self.price_tolerance: float = 0.001  # 0.1% for open=low/high detection

        # Data downloading
        self.concurrent_downloads: bool = True

        # Timezone configuration
        self.timezone = pytz.timezone('Asia/Kolkata')

        # Market hours configuration (all times in IST)
        self.market_hours: Dict = {
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

        # Default market for unknown symbols
        self.default_market: str = 'NYSE'

        # Risk management
        self.enable_eod_closure: bool = True

        # API Keys for data sources
        self.alpha_vantage_key: str = "GFAICALUQMRA3RBJ"
        self.polygon_key: str = "2elJe1MBvogj0rVlx0v4NstAGeL0VoW9"

        # Data source preference order
        self.data_sources: List[str] = ['yahoo_direct', 'yfinance', 'alpha_vantage', 'polygon']

        # File paths
        self.logs_dir: str = "logs"
        self.exports_dir: str = "exports"

    def detect_market(self, symbol: str) -> str:
        """Detect market based on symbol suffix"""
        if symbol.endswith('.NS') or symbol.endswith('.BO'):
            return 'NSE'
        elif any(symbol.endswith(suffix) for suffix in ['.L', '.TO', '.AX', '-USD']):
            return 'OTHER'
        else:
            return 'NYSE'  # Default for US stocks

    def get_valid_timeframes(self) -> List[str]:
        """Get list of valid timeframes"""
        return ['1m', '5m', '15m', '30m', '1h', '1d']

    def validate_timeframe(self, timeframe: str) -> bool:
        """Validate if timeframe is supported"""
        return timeframe in self.get_valid_timeframes()


# Global configuration instance
config = TradingConfig()