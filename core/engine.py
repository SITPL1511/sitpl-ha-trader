"""
Main trading engine implementation
"""

import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional

from config import TradingConfig
from core.models import Trade, TradeSignal, PerformanceMetrics
from data.signals import SignalGenerator, SignalFilter
from data.manager import DataManager
from data.indicators import HeikinAshiCalculator
from utils.time_manager import TimeManager
from utils.performance import PerformanceCalculator, TradeAnalyzer
from utils.logger import LoggerMixin


class TradingEngine(LoggerMixin):
    """Main trading engine that orchestrates all components"""

    def __init__(self, config: TradingConfig):
        super().__init__()
        self.config = config

        # Initialize components
        self.data_manager = DataManager(config)
        self.signal_generator = SignalGenerator(config)
        self.signal_filter = SignalFilter(config)
        self.time_manager = TimeManager(config)
        self.performance_calculator = PerformanceCalculator()
        self.trade_analyzer = TradeAnalyzer()

        # Trading state
        self.trades: List[Trade] = []
        self.open_positions: Dict[str, Trade] = {}
        self.trade_id_counter = 0

        # Tracking
        self.eod_closures = []
        self.stop_loss_hits = []

        # Ensure directories exist
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure required directories exist"""
        for directory in [self.config.logs_dir, self.config.exports_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                self.log_info(f"Created directory: {directory}")

    def run_backtest(self, start_date: str = None, end_date: str = None) -> Dict:
        """Run historical backtest"""
        start_time = datetime.now()
        self.log_info("Starting backtest with IST timezone and EOD closure logic...")

        try:
            # Download data
            data_dict = self._download_market_data()
            if not data_dict:
                return self._create_empty_result()

            # Process signals and execute trades
            all_signals = self._generate_all_signals(data_dict)
            self._execute_trading_signals(all_signals, data_dict)

            # Calculate results
            results = self._calculate_backtest_results(start_time, all_signals, data_dict)

            self.log_info(f"Backtest completed successfully in {results['execution_time']:.2f} seconds")
            return results

        except Exception as e:
            self.log_error(f"Error during backtest: {e}", exc_info=True)
            return self._create_error_result(str(e))

    def _download_market_data(self) -> Dict[str, pd.DataFrame]:
        """Download market data for all symbols"""
        download_kwargs = {
            'alpha_vantage_key': self.config.alpha_vantage_key,
            'polygon_key': self.config.polygon_key,
            'skip_validation': True
        }

        self.log_info(f"Downloading data for symbols: {self.config.symbols}")
        data_dict = self.data_manager.download_data(
            symbols=self.config.symbols,
            period="1y",
            interval=self.config.timeframe,
            **download_kwargs
        )

        if not data_dict:
            self.log_error("No data downloaded! Check your API keys or symbols.")
            return {}

        # Log data summary
        summary = self.data_manager.get_data_summary(data_dict)
        self.log_info(f"Downloaded data for {summary['total_symbols']} symbols")

        return data_dict

    def _generate_all_signals(self, data_dict: Dict[str, pd.DataFrame]) -> List[TradeSignal]:
        """Generate signals for all symbols"""
        all_signals = []

        for symbol, data in data_dict.items():
            self.log_info(f"Processing {symbol} - Market: {self.config.detect_market(symbol)}")

            # Calculate Heikin Ashi
            ha_data = HeikinAshiCalculator.calculate(data)
            if ha_data is None:
                self.log_warning(f"Failed to calculate Heikin Ashi for {symbol}")
                continue

            # Generate signals
            signals = self.signal_generator.generate_signals(ha_data, symbol)
            filtered_signals = self.signal_filter.filter_signals(signals)

            all_signals.extend(filtered_signals)
            self.log_info(f"Generated {len(filtered_signals)} filtered signals for {symbol}")

        # Sort signals by timestamp
        all_signals.sort(key=lambda x: x.timestamp)
        self.log_info(f"Total signals generated: {len(all_signals)}")

        return all_signals

    def _execute_trading_signals(self, signals: List[TradeSignal],
                                 price_data: Dict[str, pd.DataFrame]):
        """Execute all trading signals"""
        for signal in signals:
            try:
                self._process_signal(signal, price_data)
            except Exception as e:
                self.log_error(f"Error processing signal for {signal.symbol}: {e}")

        # Close remaining positions
        self._close_all_positions_at_end()

    def _process_signal(self, signal: TradeSignal, price_data: Dict[str, pd.DataFrame]):
        """Process individual trading signal"""
        symbol = signal.symbol

        if signal.signal_type in ['LONG', 'SHORT']:
            self._handle_entry_signal(signal)
        elif signal.signal_type in ['EXIT_LONG', 'EXIT_SHORT']:
            self._handle_exit_signal(signal)
        elif signal.signal_type == 'EOD_EXIT':
            self._handle_eod_signal(signal)
        elif signal.signal_type == 'STOP_LOSS':
            self._handle_stop_loss_signal(signal, price_data)

    def _handle_entry_signal(self, signal: TradeSignal):
        """Handle entry signals (LONG/SHORT)"""
        symbol = signal.symbol

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

        ist_time = self.time_manager.format_ist_timestamp(signal.timestamp)
        self.log_info(f"Opened {signal.signal_type} position for {symbol} at {signal.entry_price:.4f} [{ist_time}]")

    def _handle_exit_signal(self, signal: TradeSignal):
        """Handle exit signals"""
        symbol = signal.symbol

        if symbol in self.open_positions:
            trade = self.open_positions[symbol]

            # Check if exit matches position type
            if ((signal.signal_type == 'EXIT_LONG' and trade.side == 'LONG') or
                    (signal.signal_type == 'EXIT_SHORT' and trade.side == 'SHORT')):

                self._close_position(trade, signal, 'HA_COLOR_CHANGE')

    def _handle_eod_signal(self, signal: TradeSignal):
        """Handle end-of-day exit signals"""
        symbol = signal.symbol

        if symbol in self.open_positions:
            trade = self.open_positions[symbol]
            self._close_position(trade, signal, 'EOD_CLOSURE')
            self.eod_closures.append({
                'symbol': symbol,
                'timestamp': signal.timestamp,
                'side': trade.side
            })

    def _handle_stop_loss_signal(self, signal: TradeSignal,
                                 price_data: Dict[str, pd.DataFrame]):
        """Handle stop loss signals"""
        symbol = signal.symbol

        if symbol not in self.open_positions or symbol not in price_data:
            return

        trade = self.open_positions[symbol]
        current_data = price_data[symbol]

        # Find current bar for stop loss check
        current_bar = self._find_current_bar(signal, current_data)
        if current_bar is None:
            return

        # Check if stop loss is hit
        if self._is_stop_loss_hit(trade, current_bar):
            # Create exit signal with stop loss price
            exit_signal = TradeSignal(
                timestamp=signal.timestamp,
                symbol=symbol,
                signal_type='STOP_LOSS_EXIT',
                entry_price=trade.stop_loss,
                stop_loss=0.0000,
                ha_close=signal.ha_close,
                ha_open=signal.ha_open,
                regular_close=signal.regular_close,
                regular_open=signal.regular_open
            )

            self._close_position(trade, exit_signal, 'STOP_LOSS')
            self.stop_loss_hits.append({
                'symbol': symbol,
                'timestamp': signal.timestamp,
                'side': trade.side,
                'stop_loss_price': trade.stop_loss
            })

    def _find_current_bar(self, signal: TradeSignal, data: pd.DataFrame) -> Optional[pd.Series]:
        """Find the current bar for stop loss checking"""
        for _, row in data.iterrows():
            if 'Datetime' in data.columns:
                bar_time = row['Datetime']
            else:
                bar_time = row.name

            # Compare timestamps (handle timezone issues)
            try:
                if hasattr(bar_time, 'tz_localize') and bar_time.tz is None:
                    bar_time = bar_time.tz_localize(self.time_manager.utc_tz).astimezone(self.time_manager.ist_tz)
                elif hasattr(bar_time, 'astimezone'):
                    bar_time = bar_time.astimezone(self.time_manager.ist_tz)
            except:
                pass

            if bar_time == signal.timestamp:
                return row

        return None

    def _is_stop_loss_hit(self, trade: Trade, current_bar: pd.Series) -> bool:
        """Check if stop loss is hit for current bar"""
        if trade.side == 'LONG':
            # Long position: stop loss triggered if low breaches stop loss
            return float(current_bar['Low']) <= trade.stop_loss
        else:  # SHORT position
            # Short position: stop loss triggered if high breaches stop loss
            return float(current_bar['High']) >= trade.stop_loss

    def _close_position(self, trade: Trade, signal: TradeSignal, reason: str):
        """Close a position and calculate P&L"""
        trade.close(signal.timestamp, signal.entry_price, reason)
        del self.open_positions[trade.symbol]

        ist_time = self.time_manager.format_ist_timestamp(signal.timestamp)
        self.log_info(f"Closed {trade.side} position for {trade.symbol} at {trade.exit_price:.4f}, "
                      f"P&L: {trade.pnl:.4f}, Reason: {reason} [{ist_time}]")

    def _close_all_positions_at_end(self):
        """Force close all remaining open positions at the end of backtest"""
        if self.open_positions:
            self.log_info(f"Force closing {len(self.open_positions)} remaining open positions")

            for symbol, trade in list(self.open_positions.items()):
                trade.close(
                    exit_time=datetime.now(self.time_manager.ist_tz),
                    exit_price=trade.entry_price,  # Conservative estimate
                    exit_reason='BACKTEST_END'
                )
                trade.pnl = 0.0000  # Assuming no change for safety

                self.log_info(f"Force closed {trade.side} position for {symbol} at backtest end")

            self.open_positions.clear()

    def _calculate_backtest_results(self, start_time: datetime,
                                    all_signals: List[TradeSignal],
                                    data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate and compile backtest results"""
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # Calculate performance metrics
        performance = self.performance_calculator.calculate_performance(self.trades)

        # Generate trade analysis
        trade_analysis = self.trade_analyzer.analyze_trades(self.trades)

        # Log summary
        self.log_info(f"Total trades: {len(self.trades)}")
        self.log_info(f"EOD closures: {len(self.eod_closures)}")
        self.log_info(f"Stop loss hits: {len(self.stop_loss_hits)}")

        return {
            'trades': self.trades,
            'performance': performance,
            'trade_analysis': trade_analysis,
            'execution_time': execution_time,
            'total_signals': len(all_signals),
            'data_sources_used': list(data_dict.keys()),
            'eod_closures': len(self.eod_closures),
            'stop_loss_hits': len(self.stop_loss_hits),
            'timezone': 'Asia/Kolkata (IST)',
            'config': {
                'symbols': self.config.symbols,
                'timeframe': self.config.timeframe,
                'max_positions': self.config.max_positions,
                'eod_closure_enabled': self.config.enable_eod_closure
            }
        }

    def _create_empty_result(self) -> Dict:
        """Create empty result for failed backtest"""
        return {
            'trades': [],
            'performance': PerformanceMetrics(),
            'execution_time': 0,
            'total_signals': 0,
            'error': 'No data downloaded'
        }

    def _create_error_result(self, error_message: str) -> Dict:
        """Create error result"""
        return {
            'trades': [],
            'performance': PerformanceMetrics(),
            'execution_time': 0,
            'total_signals': 0,
            'error': error_message
        }

    def export_trades(self, filename: str = None, symbol: str = None,
                      timeframe: str = None) -> str:
        """Export trades to CSV with IST timestamps"""
        try:
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                if symbol and timeframe:
                    clean_symbol = symbol.replace('.', '_').replace(':', '_').replace('/', '_')
                    filename = f"trades_{clean_symbol}_{timeframe}_{timestamp}.csv"
                else:
                    filename = f"trades_ist_{timestamp}.csv"

            # Ensure exports directory exists
            exports_path = os.path.join(self.config.exports_dir, filename)

            trades_data = []
            for trade in self.trades:
                # Format IST timestamps
                entry_time_ist = self.time_manager.format_ist_timestamp(trade.entry_time) if trade.entry_time else None
                exit_time_ist = self.time_manager.format_ist_timestamp(trade.exit_time) if trade.exit_time else None

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
                    'status': trade.status,
                    'duration_hours': trade.duration_hours()
                })

            df = pd.DataFrame(trades_data)
            df.to_csv(exports_path, index=False)
            self.log_info(f"Trades exported to {exports_path}")
            return exports_path

        except Exception as e:
            self.log_error(f"Error exporting trades: {e}")
            return ""

    def get_market_hours_info(self) -> Dict:
        """Get market hours information for all configured markets"""
        market_info = {}

        for market, hours in self.config.market_hours.items():
            market_info[market] = {
                'open': hours['open'].strftime('%H:%M'),
                'close': hours['close'].strftime('%H:%M'),
                'eod_exit': hours['eod_exit_time'].strftime('%H:%M'),
                'spans_midnight': hours['close'] < hours['open']
            }

        return market_info

    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        return {
            'open_positions': len(self.open_positions),
            'total_trades': len(self.trades),
            'symbols': list(self.open_positions.keys()),
            'position_details': [
                {
                    'symbol': trade.symbol,
                    'side': trade.side,
                    'entry_price': trade.entry_price,
                    'entry_time': self.time_manager.format_ist_timestamp(trade.entry_time),
                    'stop_loss': trade.stop_loss
                }
                for trade in self.open_positions.values()
            ]
        }

    def reset_engine(self):
        """Reset engine state for new backtest"""
        self.trades.clear()
        self.open_positions.clear()
        self.trade_id_counter = 0
        self.eod_closures.clear()
        self.stop_loss_hits.clear()
        self.log_info("Trading engine reset")


class EngineFactory:
    """Factory for creating trading engine instances"""

    @staticmethod
    def create_engine(config: TradingConfig = None) -> TradingEngine:
        """Create a trading engine instance"""
        if config is None:
            from config import config as default_config
            config = default_config

        return TradingEngine(config)

    @staticmethod
    def create_engine_with_symbols(symbols: List[str], timeframe: str = '5m') -> TradingEngine:
        """Create engine with specific symbols and timeframe"""
        from config import config as default_config

        # Create a copy of config
        custom_config = TradingConfig()
        custom_config.symbols = symbols
        custom_config.timeframe = timeframe

        return TradingEngine(custom_config)