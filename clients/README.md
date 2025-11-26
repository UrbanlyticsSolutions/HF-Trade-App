# Gold Data Collection Workflow

This directory contains a complete standalone gold data collection workflow and supporting modules.

## Main Gold Data Collection Workflow (`gold_data_collector.py`)

**Pure data collection and processing workflow - no prediction models or technical indicators.**

### Purpose
- Collect comprehensive gold-related data from multiple sources (FMP + FRED)
- Validate and clean data quality
- Process and clean data (missing values, outliers)
- Generate basic data analysis and correlations
- Export clean data in multiple formats
- Generate detailed quality reports

### Key Features
- **Pure Data Focus**: No prediction models or technical indicators
- **Data Sources**: FMP (market data) + FRED (economic data)
- **Data Processing**: Missing data handling, outlier detection, quality validation
- **Analysis**: Correlation analysis, top indicator ranking, basic statistics
- **Export Formats**: CSV, JSON
- **Quality Control**: Completeness thresholds, validation reports

### Usage
```bash
# Basic usage with API keys
python gold_data_collector.py --fmp_key YOUR_FMP_KEY --fred_key YOUR_FRED_KEY

# With configuration file
python gold_data_collector.py --config gold_data_config.yaml --fmp_key YOUR_FMP_KEY --fred_key YOUR_FRED_KEY

# Custom parameters
python gold_data_collector.py --fmp_key YOUR_FMP_KEY --fred_key YOUR_FRED_KEY --years 10 --output_dir my_data --formats csv json
```

### Configuration
The workflow can be configured using `gold_data_config.yaml`:
- Data collection parameters (years, quality thresholds)
- Processing options (missing data strategy, outlier detection)
- Analysis options (correlation matrix, top indicators)
- Output formats and directory

## Supporting Modules

### Core Data Collection
- **`gold_comprehensive_data.py`**: Core data collection from FMP and FRED
- **`gold_range_predictor.py`**: Prediction and modeling module (separate from data collection)

### API Clients
- **`fmp_client.py`**: FMP API client
- **`fred_client.py`**: FRED API client
- **`fmp_stable_client.py`**: Alternative FMP client implementation

### Other Data Sources
- **`rss_client.py`**: RSS feed client
- **`yahoo_finance_client.py`**: Yahoo Finance API client
- **`database.py`**: Database utilities

## Data Sources

### FMP (Financial Modeling Prep)
- Gold prices (futures, ETFs, spot)
- Commodity prices (oil, silver, copper)
- Market indices (S&P 500, NASDAQ)
- Currency pairs and treasury yields
- Market volatility indicators

### FRED (Federal Reserve Economic Data)
- Fed policy indicators (Fed Funds rate, balance sheet)
- Economic indicators (GDP, inflation, employment)
- Monetary aggregates and dollar strength
- Interest rate spreads and yields

## File Structure

```
clients/
├── gold_data_collector.py             # Complete data collection workflow
├── gold_data_config.yaml              # Configuration for data collection
├── gold_comprehensive_data.py         # Core data collection module
├── gold_range_predictor.py           # Prediction and modeling module
├── fmp_client.py                     # FMP API client
├── fmp_stable_client.py              # Alternative FMP client
├── fred_client.py                    # FRED API client
├── rss_client.py                     # RSS feed client
├── yahoo_finance_client.py           # Yahoo Finance API client
├── database.py                       # Database utilities
└── README.md                         # This documentation
```

## API Keys Required

The workflow requires API keys:
- **FMP API Key**: Financial Modeling Prep (https://site.financialmodelingprep.com/)
- **FRED API Key**: Federal Reserve Economic Data (https://fred.stlouisfed.org/docs/api/api_key.html)

Set them via command line arguments or environment variables (`FMP_API_KEY`, `FRED_API_KEY`).