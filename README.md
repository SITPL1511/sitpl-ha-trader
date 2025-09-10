# Enhanced Trading Engine

A comprehensive, modular trading system with multi-source data feeds, IST timezone support, and advanced risk management features.

## Features

- 🔄 **Multi-Source Data**: Fallback system with Alpha Vantage, Polygon.io, Yahoo Finance, and yfinance
- 🌍 **IST Timezone**: Automatic conversion to Indian Standard Time
- ⏰ **Market Hours**: Intelligent market detection and EOD position closure
- 📊 **Heikin Ashi**: Advanced candlestick pattern analysis
- 🛡️ **Risk Management**: Stop loss, position sizing, and EOD closure
- 📈 **Performance Metrics**: Comprehensive analysis including Sharpe ratio
- 🏗️ **Modular Design**: Clean, maintainable code structure
- 📝 **Extensive Logging**: Detailed logging and error tracking

## Project Structure

```
trading_engine/
│
├── main.py                     # Entry point and CLI interface
├── requirements.txt            # Python dependencies
├── config.py                   # Configuration settings
│
├── core/                       # Core trading logic
│   ├── models.py              # Data models (TradeSignal, Trade)
│   ├── engine.py              # Main trading engine
│   └── signals.py             # Signal generation logic
│
├── data/                       # Data handling modules
│   ├── manager.py             # Data download manager
│   ├── sources.py             # Individual data source implementations
│   └── indicators.py          # Technical indicators (Heikin Ashi)
│
├── utils/                      # Utility modules
│   ├── time_manager.py        # Timezone and market hours handling
│   ├── logger.py              # Logging configuration
│   └── performance.py         # Performance metrics calculation
│
├── logs/                       # Log files directory
├── exports/                    # CSV exports directory
└── tests/                      # Unit tests
```

## Installation

1. Clone or download the project files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up API keys in `config.py`:
    - Alpha Vantage API key
    - Polygon.io API key

## Usage

### Basic Usage

```bash
# Use default configuration
python main.py

# Single symbol with timeframe
python main.py AAPL 5m

# Indian stock
python main.py RELIANCE.NS 15m
```

### Advanced Usage

```bash
# With API keys
python main.py GOOGL 1h --api-av YOUR_AV_KEY --api-polygon YOUR_POLYGON_KEY

# Enable concurrent downloads
python main.py MSFT 5m --concurrent

# Disable EOD closure
python main.py TSLA 15m --no-eod

# Set logging level and export detailed report
python main.py AAPL 5m --log-level DEBUG --export-report
```

### Supported Timeframes
- `1m` - 1 minute
- `5m` - 5 minutes
- `15m` - 15 minutes
- `30m` - 30 minutes
- `1h` - 1 hour
- `1d` - 1 day

## Configuration

Edit `config.py` to customize:

- **Symbols**: List of trading symbols
- **Timeframe**: Default timeframe
- **Risk Management**: Position size, risk per trade, max positions
- **Market Hours**: Trading hours for different markets
- **API Keys**: Data source API keys

## Market Hours (IST)

| Market | Open | Close | EOD Exit |
|--------|------|-------|----------|
| NSE | 09:15 | 15:30 | 15:15 |
| NYSE | 19:30 | 02:00+1 | 01:45+1 |
| NASDAQ | 19:30 | 02:00+1 | 01:45+1 |

## Signal Types

- **LONG**: Enter long position
- **SHORT**: Enter short position
- **EXIT_LONG**: Exit long on HA color change
- **EXIT_SHORT**: Exit short on HA color change
- **EOD_EXIT**: Exit before market close
- **STOP_LOSS**: Exit on stop loss breach

## Performance Metrics

- Win Rate and P&L analysis
- Sharpe Ratio for risk-adjusted returns
- Maximum Drawdown
- Trade duration statistics
- Exit reason breakdown
- Symbol-wise performance

## Output Files

### Trade Export (CSV)
- Individual trade details with IST timestamps
- Entry/exit prices and reasons
- P&L and duration analysis

### Log Files
- Detailed execution logs
- Error tracking and debugging
- Performance timing

### Performance Reports
- Comprehensive performance analysis
- Risk metrics and statistics
- Trade pattern analysis

## Risk Management Features

1. **Stop Loss**: Automatic position closure on price breach
2. **EOD Closure**: Positions closed before market close to avoid gaps
3. **Position Limits**: Maximum concurrent positions
4. **Market Hours**: Only trade during market hours
5. **Data Validation**: Comprehensive data quality checks

## Architecture Benefits

### Modularity
- Separated concerns for easy maintenance
- Pluggable data sources
- Configurable components

### Error Handling
- Graceful degradation with fallback data sources
- Comprehensive error logging
- Input validation

### Performance
- Vectorized calculations
- Concurrent data downloads
- Efficient memory usage

### Extensibility
- Easy to add new data sources
- Pluggable signal generators
- Configurable risk management

## Development

### Adding New Data Sources
1. Create new class inheriting from `DataSourceBase` in `data/sources.py`
2. Implement the `download` method
3. Register in `DataSourceFactory`

### Adding New Indicators
1. Add indicator calculation in `data/indicators.py`
2. Use vectorized operations for performance
3. Include proper error handling

### Adding New Signal Types
1. Extend signal generation logic in `core/signals.py`
2. Update `TradeSignal` model if needed
3. Add processing logic in `core/engine.py`

## Troubleshooting

### Common Issues

1. **No data downloaded**: Check API keys and symbol validity
2. **Timezone errors**: Ensure proper timezone handling in data
3. **Memory issues**: Reduce data lookback period or use smaller timeframes
4. **Rate limiting**: Enable delays between API calls

### Debug Mode
```bash
python main.py SYMBOL 5m --log-level DEBUG
```

### Validation
- All symbols are validated before processing
- Data quality checks prevent bad data
- Market hours validation ensures proper timing

## Contributing

1. Follow the modular structure
2. Add comprehensive error handling
3. Include logging for debugging
4. Write unit tests for new features
5. Update documentation

## License

This project is provided as-is for educational and research purposes.

## Disclaimer

This trading engine is for educational purposes only. Past performance does not guarantee future results. Always test thoroughly before using with real capital.