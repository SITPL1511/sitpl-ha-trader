"""
Performance metrics calculation utilities
"""

import numpy as np
import pandas as pd
from typing import List, Dict
import logging

from core.models import Trade, PerformanceMetrics
from utils.logger import LoggerMixin


class PerformanceCalculator(LoggerMixin):
    """Calculate comprehensive performance metrics"""

    def __init__(self):
        super().__init__()

    def calculate_performance(self, trades: List[Trade]) -> PerformanceMetrics:
        """Calculate performance metrics from list of trades"""
        try:
            closed_trades = [t for t in trades if t.is_closed()]

            if not closed_trades:
                self.log_warning("No closed trades found for performance calculation")
                return PerformanceMetrics()

            # Extract P&L values
            pnls = [t.pnl for t in closed_trades if t.pnl is not None]

            if not pnls:
                self.log_warning("No P&L values found in closed trades")
                return PerformanceMetrics(total_trades=len(closed_trades))

            # Basic metrics
            metrics = PerformanceMetrics()
            metrics.total_trades = len(closed_trades)

            # Win/Loss analysis
            win_trades = [p for p in pnls if p > 0]
            lose_trades = [p for p in pnls if p < 0]

            metrics.winning_trades = len(win_trades)
            metrics.losing_trades = len(lose_trades)
            metrics.win_rate = round(len(win_trades) / len(closed_trades), 4) if closed_trades else 0.0

            # P&L metrics
            metrics.total_pnl = round(sum(pnls), 4)
            metrics.average_win = round(np.mean(win_trades), 4) if win_trades else 0.0
            metrics.average_loss = round(np.mean(lose_trades), 4) if lose_trades else 0.0
            metrics.max_win = round(max(pnls), 4) if pnls else 0.0
            metrics.max_loss = round(min(pnls), 4) if pnls else 0.0

            # Profit factor
            total_wins = sum(win_trades) if win_trades else 0
            total_losses = abs(sum(lose_trades)) if lose_trades else 0
            metrics.profit_factor = round(total_wins / total_losses, 4) if total_losses > 0 else float('inf')

            # Trade duration analysis
            durations = [t.duration_hours() for t in closed_trades if t.duration_hours() is not None]
            metrics.average_trade_duration_hours = round(np.mean(durations), 4) if durations else 0.0

            # Exit reason analysis
            metrics.eod_closures = sum(1 for t in closed_trades if t.exit_reason == 'EOD_CLOSURE')
            metrics.ha_color_exits = sum(1 for t in closed_trades if t.exit_reason == 'HA_COLOR_CHANGE')
            metrics.stop_loss_exits = sum(1 for t in closed_trades if t.exit_reason == 'STOP_LOSS')

            # Risk metrics
            metrics.sharpe_ratio = self._calculate_sharpe_ratio(pnls)
            metrics.max_drawdown = self._calculate_max_drawdown(pnls)

            self.log_info(f"Calculated performance metrics for {len(closed_trades)} trades")
            return metrics

        except Exception as e:
            self.log_error(f"Error calculating performance metrics: {e}", exc_info=True)
            return PerformanceMetrics()

    def _calculate_sharpe_ratio(self, pnls: List[float], risk_free_rate: float = 0.05) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(pnls) <= 1:
                return 0.0

            daily_pnl_std = np.std(pnls, ddof=1)
            mean_daily_pnl = np.mean(pnls)

            if daily_pnl_std > 0:
                # Annualized Sharpe ratio (assuming 252 trading days per year)
                sharpe_ratio = (mean_daily_pnl * 252 - risk_free_rate) / (daily_pnl_std * np.sqrt(252))
                return round(sharpe_ratio, 4)

            return 0.0

        except Exception as e:
            self.log_error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def _calculate_max_drawdown(self, pnls: List[float]) -> float:
        """Calculate maximum drawdown"""
        try:
            if not pnls:
                return 0.0

            cumulative_pnl = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = running_max - cumulative_pnl
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0

            return round(max_drawdown, 4)

        except Exception as e:
            self.log_error(f"Error calculating max drawdown: {e}")
            return 0.0


class TradeAnalyzer(LoggerMixin):
    """Analyze trading patterns and behavior"""

    def analyze_trades(self, trades: List[Trade]) -> Dict:
        """Comprehensive trade analysis"""
        try:
            analysis = {
                'total_trades': len(trades),
                'open_trades': len([t for t in trades if t.is_open()]),
                'closed_trades': len([t for t in trades if t.is_closed()]),
                'by_symbol': {},
                'by_side': {'LONG': 0, 'SHORT': 0},
                'by_exit_reason': {},
                'duration_stats': {},
                'pnl_distribution': {}
            }

            closed_trades = [t for t in trades if t.is_closed()]

            # Analysis by symbol
            for trade in trades:
                symbol = trade.symbol
                if symbol not in analysis['by_symbol']:
                    analysis['by_symbol'][symbol] = {'total': 0, 'open': 0, 'closed': 0}

                analysis['by_symbol'][symbol]['total'] += 1
                if trade.is_open():
                    analysis['by_symbol'][symbol]['open'] += 1
                else:
                    analysis['by_symbol'][symbol]['closed'] += 1

            # Analysis by side
            for trade in trades:
                analysis['by_side'][trade.side] += 1

            # Analysis by exit reason
            for trade in closed_trades:
                if trade.exit_reason:
                    reason = trade.exit_reason
                    analysis['by_exit_reason'][reason] = analysis['by_exit_reason'].get(reason, 0) + 1

            # Duration statistics
            durations = [t.duration_hours() for t in closed_trades if t.duration_hours() is not None]
            if durations:
                analysis['duration_stats'] = {
                    'mean': round(np.mean(durations), 2),
                    'median': round(np.median(durations), 2),
                    'min': round(min(durations), 2),
                    'max': round(max(durations), 2),
                    'std': round(np.std(durations), 2)
                }

            # P&L distribution
            pnls = [t.pnl for t in closed_trades if t.pnl is not None]
            if pnls:
                analysis['pnl_distribution'] = {
                    'mean': round(np.mean(pnls), 4),
                    'median': round(np.median(pnls), 4),
                    'min': round(min(pnls), 4),
                    'max': round(max(pnls), 4),
                    'std': round(np.std(pnls), 4),
                    'positive_count': len([p for p in pnls if p > 0]),
                    'negative_count': len([p for p in pnls if p < 0]),
                    'zero_count': len([p for p in pnls if p == 0])
                }

            return analysis

        except Exception as e:
            self.log_error(f"Error analyzing trades: {e}", exc_info=True)
            return {}


class ReportGenerator(LoggerMixin):
    """Generate performance reports"""

    def generate_summary_report(self, metrics: PerformanceMetrics,
                                trade_analysis: Dict = None) -> str:
        """Generate a summary performance report"""
        try:
            report_lines = [
                "=" * 50,
                "TRADING PERFORMANCE SUMMARY",
                "=" * 50,
                f"Total Trades: {metrics.total_trades}",
                f"Winning Trades: {metrics.winning_trades}",
                f"Losing Trades: {metrics.losing_trades}",
                f"Win Rate: {metrics.win_rate:.4f} ({metrics.win_rate*100:.2f}%)",
                "",
                f"Total P&L: ${metrics.total_pnl:.4f}",
                f"Average Win: ${metrics.average_win:.4f}",
                f"Average Loss: ${metrics.average_loss:.4f}",
                f"Max Win: ${metrics.max_win:.4f}",
                f"Max Loss: ${metrics.max_loss:.4f}",
                "",
                f"Profit Factor: {metrics.profit_factor:.4f}",
                f"Sharpe Ratio: {metrics.sharpe_ratio:.4f}",
                f"Max Drawdown: ${metrics.max_drawdown:.4f}",
                f"Avg Trade Duration: {metrics.average_trade_duration_hours:.2f} hours",
                "",
                "EXIT BREAKDOWN:",
                f"  EOD Closures: {metrics.eod_closures}",
                f"  HA Color Exits: {metrics.ha_color_exits}",
                f"  Stop Loss Exits: {metrics.stop_loss_exits}",
                "",
                f"Timezone: {metrics.timezone}",
                "=" * 50
            ]

            if trade_analysis:
                report_lines.extend([
                    "",
                    "TRADE ANALYSIS:",
                    f"  Open Positions: {trade_analysis.get('open_trades', 0)}",
                    f"  Symbols Traded: {len(trade_analysis.get('by_symbol', {}))}",
                    f"  Long Trades: {trade_analysis.get('by_side', {}).get('LONG', 0)}",
                    f"  Short Trades: {trade_analysis.get('by_side', {}).get('SHORT', 0)}"
                ])

            return "\n".join(report_lines)

        except Exception as e:
            self.log_error(f"Error generating summary report: {e}")
            return "Error generating report"

    def export_detailed_report(self, metrics: PerformanceMetrics,
                               trades: List[Trade], filename: str = None) -> str:
        """Export detailed performance report to file"""
        try:
            if not filename:
                timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                filename = f"performance_report_{timestamp}.txt"

            # Generate comprehensive report
            report_content = self.generate_summary_report(metrics)

            # Add trade details
            trade_analyzer = TradeAnalyzer()
            trade_analysis = trade_analyzer.analyze_trades(trades)

            with open(filename, 'w') as f:
                f.write(report_content)
                f.write("\n\n")
                f.write("DETAILED TRADE ANALYSIS:\n")
                f.write("=" * 50 + "\n")

                # Write trade analysis
                for key, value in trade_analysis.items():
                    f.write(f"{key}: {value}\n")

            self.log_info(f"Detailed report exported to: {filename}")
            return filename

        except Exception as e:
            self.log_error(f"Error exporting detailed report: {e}")
            return ""