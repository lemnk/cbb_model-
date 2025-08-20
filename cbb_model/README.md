# CBB Betting ML System

A comprehensive machine learning system for College Basketball (NCAA) betting analysis and prediction.

## 🎯 Overview

This system is designed to collect, process, and analyze NCAA basketball data to build predictive models for betting outcomes. The project is structured in phases:

- **Phase 1: Data Infrastructure** (Current) - Data collection, storage, and ETL
- **Phase 2: Feature Engineering** - Advanced feature creation and selection
- **Phase 3: Model Training** - ML model development and training

## 🏗️ Architecture

```
cbb_model/
│── data/                    # Data storage
│   ├── raw/                # Raw scraped data
│   ├── processed/          # Cleaned and merged data
│   └── backup/             # Data backups
│── notebooks/              # Jupyter notebooks for exploration
│── src/                    # Core source code
│   ├── scrape_games.py     # NCAA games data scraper
│   ├── scrape_odds.py      # Betting odds scraper
│   ├── db.py              # Database management
│   ├── etl.py             # Data processing pipeline
│   ├── features.py         # Feature engineering (Phase 2)
│   ├── train.py            # ML training (Phase 3)
│   └── utils.py            # Shared utilities
│── requirements.txt         # Python dependencies
│── config.yaml             # Configuration file
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- PostgreSQL database
- Internet connection for data scraping

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd cbb_model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

1. Copy the configuration template:
```bash
cp config.yaml config_local.yaml
```

2. Edit `config_local.yaml` with your settings:
```yaml
database:
  host: localhost
  port: 5432
  name: cbb_betting
  user: your_username
  password: your_password

apis:
  pinnacle:
    api_key: your_pinnacle_api_key  # Optional
```

### 4. Database Setup

```bash
# Create PostgreSQL database
createdb cbb_betting

# Run database setup (this will create tables)
python -m src.db
```

### 5. Data Collection

```bash
# Collect NCAA games data (test with limited teams first)
python -m src.scrape_games

# Collect betting odds data
python -m src.scrape_odds

# Run ETL pipeline
python -m src.etl
```

## 📊 Data Sources

### NCAA Basketball Data
- **Source**: [Sports Reference](https://www.sports-reference.com/cbb/)
- **Data**: Game results, scores, team stats, player performance
- **Coverage**: Historical seasons, current season updates

### Betting Odds Data
- **Pinnacle Sports**: Professional odds and line movements
- **DraftKings**: Retail sportsbook odds
- **Data**: Moneyline, spreads, totals, opening/closing lines

## 🗄️ Database Schema

### Core Tables

#### `games`
- Game results and basic statistics
- Team performance metrics
- Game metadata (date, location, attendance)

#### `odds`
- Betting odds from multiple sportsbooks
- Line movements over time
- Opening vs. closing odds

#### `players`
- Individual player statistics
- Per-game performance metrics
- Advanced stats (efficiency, pace)

#### `teams`
- Team information and metadata
- Conference and division details
- Venue information

#### `games_odds`
- Merged view combining games and odds
- Ready for feature engineering
- Optimized for ML pipeline

## 🔧 Usage Examples

### Basic Data Collection

```python
from src.scrape_games import create_games_scraper
from src.scrape_odds import create_odds_collector

# Scrape games data
games_scraper = create_games_scraper()
result = games_scraper.scrape_and_save_season(2024, max_teams=10)
print(f"Scraped {result['games_count']} games")

# Collect odds data
odds_collector = create_odds_collector()
result = odds_collector.collect_and_save_odds(days_back=30)
print(f"Collected {result['odds_count']} odds records")
```

### ETL Pipeline

```python
from src.etl import create_etl_processor

# Run complete ETL pipeline
etl_processor = create_etl_processor()
result = etl_processor.run_full_etl_pipeline(season=2024, days_back=30)

if result['success']:
    print(f"ETL completed: {result['merged_count']} merged records")
    print(f"Features created: {result['features_count']} records")
```

### Database Operations

```python
from src.db import create_database_manager

# Database operations
db_manager = create_database_manager()
db_manager.create_tables()

# Query data
query = "SELECT * FROM games WHERE season = 2024 LIMIT 10"
results = db_manager.execute_query(query)
print(results)
```

## 📈 Data Flow

1. **Data Collection**
   - Scrape NCAA games from Sports Reference
   - Collect odds from sportsbooks
   - Store raw data in CSV files

2. **Data Processing**
   - Clean and normalize raw data
   - Merge games and odds datasets
   - Validate data quality

3. **Feature Engineering** (Phase 2)
   - Create rolling averages and trends
   - Generate team performance metrics
   - Build market efficiency indicators

4. **Model Training** (Phase 3)
   - Prepare training datasets
   - Train and validate models
   - Evaluate performance metrics

## ⚙️ Configuration

### Key Configuration Options

```yaml
# Data collection settings
data_collection:
  start_season: 2020
  end_season: 2024
  odds:
    days_back: 30
    update_frequency_hours: 6

# Feature engineering
features:
  rolling_windows: [3, 5, 10, 20]
  advanced_stats:
    - pace
    - efficiency
    - strength_of_schedule

# Model training
model:
  test_size: 0.2
  validation_size: 0.1
  cross_validation_folds: 5
```

## 🔒 Security & Ethics

### Data Usage
- Respect website terms of service
- Implement appropriate rate limiting
- Use data responsibly and ethically

### API Keys
- Store API keys in environment variables
- Never commit sensitive credentials
- Use separate config files for local development

## 🧪 Testing

### Unit Tests
```bash
# Run tests (when implemented)
python -m pytest tests/
```

### Integration Tests
```bash
# Test data pipeline
python -m src.etl
python -m src.scrape_games
python -m src.scrape_odds
```

## 📝 Development

### Code Style
- Follow PEP 8 guidelines
- Use type hints throughout
- Comprehensive docstrings
- Logging for debugging

### Adding New Features
1. Create feature branch
2. Implement functionality
3. Add tests
4. Update documentation
5. Submit pull request

## 🚨 Troubleshooting

### Common Issues

#### Database Connection
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection
psql -h localhost -U your_username -d cbb_betting
```

#### Scraping Issues
- Check internet connection
- Verify rate limiting settings
- Review website structure changes
- Check for CAPTCHA/anti-bot measures

#### Data Quality
- Validate CSV file formats
- Check for missing columns
- Review data type conversions
- Monitor for duplicate records

## 📚 API Reference

### Core Classes

- `NCAAGamesScraper`: NCAA data collection
- `OddsDataCollector`: Betting odds collection
- `DatabaseManager`: Database operations
- `CBBDataETL`: Data processing pipeline
- `CBBFeatureEngineer`: Feature engineering (Phase 2)
- `CBBModelTrainer`: Model training (Phase 3)

### Utility Functions

- `ConfigManager`: Configuration management
- `setup_logging`: Logging configuration
- `safe_filename`: File naming utilities
- `validate_dataframe`: Data validation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Sports Reference for NCAA basketball data
- Pinnacle Sports and DraftKings for odds data
- Open source community for tools and libraries

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review troubleshooting section

---

**Note**: This is Phase 1 (Data Infrastructure). Feature engineering and model training will be implemented in subsequent phases.