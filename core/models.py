"""
Core data models for the trading engine
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class TradeSignal:
    """Trade signal data structure"""
    timestamp: datetime
    symbol: str
    signal_type: str  # 'LONG', 'SHORT', 'EXIT_LONG', 'EXIT_SHORT', 'EOD_EXIT', 'STOP_LOSS'
    entry_price: float
    stop_loss: float
    ha_close: float
    ha_open: float
    regular_close: float
    regular_open: float

    def __post_init__(self):
        """Validate signal data after initialization"""
        valid_signal_types = [
            'LONG', 'SHORT', 'EXIT_LONG', 'EXIT_SHORT',
            'EOD_EXIT', 'STOP_LOSS', 'STOP_LOSS_EXIT'
        ]
        if self.signal_type not in valid_signal_types:
            raise ValueError(f"Invalid signal type: {self.signal_type}")

    def is_entry_signal(self) -> bool:
        """Check if this is an entry signal"""
        return self.signal_type in ['LONG', 'SHORT']

    def is_exit_signal(self) -> bool:
        """Check if this is an exit signal"""
        return self.signal_type in ['EXIT_LONG', 'EXIT_SHORT', 'EOD_EXIT', 'STOP_LOSS_EXIT']


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

    def __post_init__(self):
        """Validate trade data after initialization"""
        valid_sides = ['LONG', 'SHORT']
        if self.side not in valid_sides:
            raise ValueError(f"Invalid trade side: {self.side}")

        valid_statuses = ['OPEN', 'CLOSED']
        if self.status not in valid_statuses:
            raise ValueError(f"Invalid trade status: {self.status}")

    def is_open(self) -> bool:
        """Check if trade is still open"""
        return self.status == 'OPEN'

    def is_closed(self) -> bool:
        """Check if trade is closed"""
        return self.status == 'CLOSED'

    def duration_hours(self) -> Optional[float]:
        """Calculate trade duration in hours"""
        if self.entry_time and self.exit_time:
            duration = self.exit_time - self.entry_time
            return duration.total_seconds() / 3600
        return None

    def close(self, exit_time: datetime, exit_price: float, exit_reason: str) -> None:
        """Close the trade and calculate P&L"""
        self.exit_time = exit_time
        self.exit_price = round(exit_price, 4)
        self.exit_reason = exit_reason
        self.status = 'CLOSED'

        # Calculate P&L
        if self.side == 'LONG':
            self.pnl = round(self.exit_price - self.entry_price, 4)
        else:  # SHORT
            self.pnl = round(self.entry_price - self.exit_price, 4)


@dataclass
class MarketHours:
    """Market hours configuration"""
    open_time: datetime.time
    close_time: datetime.time
    eod_exit_time: datetime.time
    timezone: str = 'Asia/Kolkata'

    def spans_midnight(self) -> bool:
        """Check if market hours span across midnight"""
        return self.close_time < self.open_time


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    profit_factor: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    average_trade_duration_hours: float = 0.0
    eod_closures: int = 0
    ha_color_exits: int = 0
    stop_loss_exits: int = 0
    timezone: str = 'Asia/Kolkata (IST)'

    def to_dict(self) -> dict:
        """Convert to dictionary for easy serialization"""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'total_pnl': self.total_pnl,
            'average_win': self.average_win,
            'average_loss': self.average_loss,
            'profit_factor': self.profit_factor,
            'max_win': self.max_win,
            'max_loss': self.max_loss,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'average_trade_duration_hours': self.average_trade_duration_hours,
            'eod_closures': self.eod_closures,
            'ha_color_exits': self.ha_color_exits,
            'stop_loss_exits': self.stop_loss_exits,
            'timezone': self.timezone
        }