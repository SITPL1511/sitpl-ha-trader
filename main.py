#!/usr/bin/env python3
"""
Enhanced Trading Engine - Main Entry Point
A comprehensive trading system with multi-source data, IST timezone support, and EOD logic
"""

import sys
import time
import argparse
from typing import Optional

from config import TradingConfig
from core.engine import TradingEngine, EngineFactory
from utils.logger import trading_logger
from utils.performance import ReportGenerator


def parse_command_line_args():
    """Parse command line arguments for symbol and timeframe"""
    parser = argparse.ArgumentParser(
        description='Enhanced Trading Engine with IST & EOD Logic',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py MAZDOCK.NS 15m
  python main.py AAPL 5m
  python main.py GOOGL 1h
  python main.py  (uses default symbols and timeframe from config)

Supported timeframes: 1m, 5m, 15m, 30m, 1h, 1d

Features:
  ✓ Multi-source data downloader with fallback
  ✓ Automatic timezone conversion to IST
  ✓ Market hours detection and validation
  ✓ End-of-Day position closure to avoid gap risks
  ✓ Stop loss management
  ✓ Comprehensive performance metrics
  ✓ Modular, maintainable code structure
        """
    )

    parser.add_argument('symbol', nargs='?', default=None,
                        help='Trading symbol (e.g., MAZDOCK.NS, AAPL, GOOGL)')
    parser.add_argument('timeframe', nargs='?', default=None,
                        help='Timeframe (1m, 5m, 15m, 30m, 1h, 1d)')

    # Optional arguments
    parser.add_argument('--api-av', dest='alpha_vantage_key', default=None,
                        help='Alpha Vantage API key')
    parser.add_argument('--api-polygon', dest='polygon_key', default=None,
                        help='Polygon.io API key')
    parser.add_argument('--concurrent', action='store_true', default=False,
                        help='Enable concurrent downloads')
    parser.add_argument('--no-eod', action='store_true', default=False,
                        help='Disable EOD closure')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Set logging level')
    parser.add_argument('--export-report', action='store_true', default=False,
                        help='Export detailed performance report')

    return parser.parse_args()


def configure_from_args(config: TradingConfig, args) -> TradingConfig:
    """Configure trading config from command line arguments"""

    # Override symbol if provided
    if args.symbol:
        config.symbols = [args.symbol]
        print(f"Using symbol from command line: {args.symbol}")

    # Override timeframe if provided and valid
    if args.timeframe:
        if config.validate_timeframe(args.timeframe):
            config.timeframe = args.timeframe
            print(f"Using timeframe from command line: {args.timeframe}")
        else:
            valid_timeframes = config.get_valid_timeframes()
            print(f"Invalid timeframe '{args.timeframe}'. Valid options: {', '.join(valid_timeframes)}")
            print("Using default timeframe from config.")

    # Set API keys if provided
    if args.alpha_vantage_key:
        config.alpha_vantage_key = args.alpha_vantage_key
        print("Alpha Vantage API key updated from command line")

    if args.polygon_key:
        config.polygon_key = args.polygon_key
        print("Polygon.io API key updated from command line")

    # Set other options
    config.concurrent_downloads = args.concurrent
    config.enable_eod_closure = not args.no_eod

    return config


def print_startup_banner(config: TradingConfig):
    """Print startup information"""
    print("=" * 60)
    print("ENHANCED TRADING ENGINE WITH IST & EOD LOGIC")
    print("=" * 60)
    print("Features:")
    print("✓ Multi-source data downloader with fallback")
    print("✓ Automatic timezone conversion to IST")
    print("✓ Market hours detection and validation")
    print("✓ End-of-Day position closure to avoid gap risks")
    print("✓ Stop loss management")
    print("✓ Separate tracking of EOD vs HA-based exits")
    print("✓ Sharpe Ratio calculation in performance metrics")
    print("✓ Command line support for symbol and timeframe")
    print("✓ Modular, maintainable code structure")
    print()

    print(f"Configuration:")
    print(f"  Symbol(s): {config.symbols}")
    print(f"  Timeframe: {config.timeframe}")
    print(f"  Concurrent Downloads: {config.concurrent_downloads}")
    print(f"  EOD Closure: {config.enable_eod_closure}")
    print(f"  Max Positions: {config.max_positions}")
    print(f"  Risk per Trade: {config.risk_per_trade * 100:.1f}%")
    print()


def print_market_hours_info(engine: TradingEngine):
    """Print market hours configuration"""
    print("=== CONFIGURED MARKET HOURS (IST) ===")
    market_info = engine.get_market_hours_info()

    for market, info in market_info.items():
        print(f"{market}:")
        print(f"  Open: {info['open']}")
        print(f"  Close: {info['close']}")
        print(f"  EOD Exit: {info['eod_exit']}")
        if info['spans_midnight']:
            print(f"  Note: Market spans midnight")
        print()


def print_results_summary(results: dict, config: TradingConfig):
    """Print backtest results summary"""
    performance = results.get('performance')
    if not performance:
        print("No performance data available")
        return

    primary_symbol = config.symbols[0] if config.symbols else "UNKNOWN"

    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Symbol: {primary_symbol}")
    print(f"Timeframe: {config.timeframe}")
    print(f"Timezone: {results.get('timezone', 'UTC')}")
    print(f"Data Sources Used: {results.get('data_sources_used', [])}")
    print(f"Execution Time: {results.get('execution_time', 0):.2f} seconds")
    print(f"Total Signals: {results.get('total_signals', 0)}")
    print(f"Total Trades: {performance.total_trades}")

    if performance.total_trades > 0:
        print(f"Win Rate: {performance.win_rate:.4f} ({performance.win_rate*100:.2f}%)")
        print(f"Total P&L: ${performance.total_pnl:.4f}")
        print(f"Max Drawdown: ${performance.max_drawdown:.4f}")
        print(f"Sharpe Ratio: {performance.sharpe_ratio:.4f}")
        print(f"Profit Factor: {performance.profit_factor:.4f}")
        print(f"Average Win: ${performance.average_win:.4f}")
        print(f"Average Loss: ${performance.average_loss:.4f}")
        print(f"Average Trade Duration: {performance.average_trade_duration_hours:.4f} hours")
        print(f"EOD Closures: {performance.eod_closures}")
        print(f"HA Color Change Exits: {performance.ha_color_exits}")
        print(f"Stop Loss Exits: {performance.stop_loss_exits}")
    else:
        print("No completed trades found.")


def print_risk_management_info():
    """Print risk management features"""
    print("\n=== RISK MANAGEMENT FEATURES ===")
    print("✓ Stop Loss: Positions closed when SL price is breached")
    print("✓ EOD Closure: Positions closed before market close")
    print("✓ HA Signal Exits: Positions closed on Heikin Ashi color change")
    print("✓ Configurable EOD exit time per market")
    print("✓ Separate tracking of all exit types")
    print("✓ Sharpe Ratio for risk-adjusted performance measurement")
    print("✓ Position size and risk management controls")


def main() -> Optional[TradingEngine]:
    """Main execution function"""
    # Parse command line arguments
    args = parse_command_line_args()

    # Initialize configuration
    config = TradingConfig()
    config = configure_from_args(config, args)

    # Setup logging
    log_level_map = {
        'DEBUG': 10, 'INFO': 20, 'WARNING': 30, 'ERROR': 40
    }
    log_level = log_level_map.get(args.log_level, 20)

    primary_symbol = config.symbols[0] if config.symbols else "UNKNOWN"
    log_file = trading_logger.setup_logging(primary_symbol, config.timeframe, log_level)

    # Print startup information
    print_startup_banner(config)

    try:
        # Initialize trading engine
        engine = EngineFactory.create_engine(config)

        # Print market hours configuration
        print_market_hours_info(engine)

        # Run backtest
        print("Starting backtest...")
        results = engine.run_backtest()

        # Handle errors
        if 'error' in results:
            print(f"Error during backtest: {results['error']}")
            return None

        # Print results
        print_results_summary(results, config)

        # Export trades if any exist
        if len(engine.trades) > 0:
            csv_filename = engine.export_trades(symbol=primary_symbol, timeframe=config.timeframe)
            print(f"\nTrade history exported to: {csv_filename}")

            # Export detailed report if requested
            if args.export_report:
                report_generator = ReportGenerator()
                report_file = report_generator.export_detailed_report(
                    results['performance'],
                    engine.trades
                )
                if report_file:
                    print(f"Detailed performance report exported to: {report_file}")

        # Print risk management info
        print_risk_management_info()

        print(f"\nLog file: {log_file}")

        return engine

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Performance timing
    start_time = time.time()

    # Run main function
    engine = main()

    # Print total execution time
    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nTotal execution time: {total_time:.2f} seconds")

    if engine:
        print("Enhanced trading engine completed successfully!")
        sys.exit(0)
    else:
        print("Trading engine encountered errors.")
        sys.exit(1)