# Phase 2: Feature Engineering - HARDENED IMPLEMENTATION

## Overview
Phase 2 implements a **hardened, production-ready feature engineering pipeline** for the CBB Betting ML System. This phase transforms raw data from Phase 1 into model-ready features while preventing data leakage, ensuring schema integrity, and maintaining data quality.

## 🛡️ Hardening Patches Applied

### 1. Schema & Key Integrity
- **`validate_keys()` function**: Validates required keys exist and contain no nulls before merges
- **Team name standardization**: Prevents merge mismatches due to casing/whitespace differences
- **Merge verification**: All joins use `game_id` consistently with validation checks
- **Duplicate prevention**: Automatic detection and removal of duplicate `game_id` entries

### 2. Data Leakage Prevention
- **Time ordering enforcement**: All rolling/stat features computed in chronological order
- **Pre-game data validation**: Market features only use odds data from before game start
- **Future data guards**: Rest days, travel calculations only use past game information
- **Rolling window safety**: Explicit time sorting before applying rolling operations

### 3. Safe Normalization & Scaling
- **`Normalizer` class**: Prevents test data leakage by fitting on training data only
- **Train/test split**: Uses first 70% of data for fitting normalizers
- **Consistent scaling**: All GSI components normalized to [0,1] before weighting
- **Fallback handling**: Graceful degradation when normalizers can't be fitted

### 4. Missing Data & Edge Cases
- **`safe_fill()` function**: Consistent missing value handling across all feature modules
- **Graceful degradation**: Pipeline continues even with missing injury reports or odds
- **Rolling feature safety**: Explicit NaN handling for edge cases (first few games)
- **Fallback defaults**: Sensible defaults when data is unavailable

### 5. Pipeline Validation
- **Input validation**: Comprehensive data validation before processing
- **Merge verification**: Each merge operation validated for success
- **Final checks**: Comprehensive validation of final feature set
- **Error reporting**: Clear error messages and warnings for debugging

## 🏗️ Architecture

### Core Components
```
src/features/
├── __init__.py              # Package initialization
├── feature_utils.py         # Validation, normalization, utilities
├── team_features.py         # Team-level efficiency and consistency
├── player_features.py       # Player availability and bench depth
├── dynamic_features.py      # Situational and contextual factors
├── market_features.py       # Market efficiency and CLV
└── feature_pipeline.py      # Orchestration and GSI computation
```

### Feature Categories

#### Team Features (`TeamFeatures`)
- **Efficiency Metrics**: Offensive/defensive efficiency, pace, consistency
- **Home/Away Splits**: Win percentages, scoring differentials, court advantage
- **Consistency Analysis**: Rolling averages (3, 5, 10 games), volatility scoring
- **Strength of Schedule**: Opponent quality, SOS-adjusted efficiency

#### Player Features (`PlayerFeatures`)
- **Injury Analysis**: Injury flags, severity scoring, recency factors
- **Foul Trouble**: Foul rates, projected minutes lost, risk assessment
- **Bench Depth**: Bench contribution, utilization rates, sixth man indicators
- **Availability Metrics**: Minutes distribution, rotation depth, substitution patterns

#### Dynamic Features (`DynamicFeatures`)
- **Streak Analysis**: Win/loss streaks, momentum scoring, streak categories
- **Rest Management**: Days since last game, fatigue indexing, rest quality
- **Travel Factors**: Distance calculations, time zone changes, travel impact
- **Situational Context**: Rivalry games, conference games, tournament indicators

#### Market Features (`MarketFeatures`)
- **Line Movement**: Spread/total movement, magnitude analysis, movement velocity
- **Implied Probability**: Moneyline conversion, market edge calculation
- **Market Efficiency**: Efficiency scoring, line stability, confidence metrics
- **Closing Line Value**: CLV calculation, overlay thresholds, edge direction

## 🎯 Game Strength Index (GSI) Formula

The GSI is computed using the **exact formula** specified in the requirements:

```
GSI = 0.35 × normalize(team_efficiency) + 
      0.25 × normalize(player_availability) + 
      0.20 × normalize(dynamic_factors) + 
      0.20 × normalize(market_signals)
```

### Component Details
- **Team Efficiency (35%)**: Combined offensive/defensive efficiency, normalized to [0,1]
- **Player Availability (25%)**: Inverse of injury impact, normalized to [0,1]
- **Dynamic Factors (20%)**: Rest quality, travel impact, normalized to [0,1]
- **Market Signals (20%)**: Market efficiency inverse, normalized to [0,1]

### Normalization Safety
- **Training data fitting**: Normalizers fitted on first 70% of data
- **Consistent scaling**: All components scaled to [0,1] range
- **Leakage prevention**: No test data used for fitting parameters

## 🔧 Technical Implementation

### Key Functions

#### `validate_keys(df, key="game_id", df_name="DataFrame")`
- Validates required key columns exist and contain no nulls
- Called before every merge operation
- Provides clear error messages for debugging

#### `ensure_time_order(df, date_col="date", team_col="team")`
- Sorts DataFrame by team and date to prevent data leakage
- Applied before all rolling window calculations
- Ensures chronological ordering for time-series features

#### `Normalizer` class
- Supports minmax, zscore, and robust normalization
- Fits on training data only to prevent leakage
- Provides transform method for consistent scaling

#### `safe_fill(df, col, fill_value=0)`
- Safely fills missing values or creates columns if missing
- Consistent missing value handling across all modules
- Prevents pipeline crashes due to missing data

### Pipeline Flow
1. **Input Validation**: Validate all required keys and data integrity
2. **Normalizer Fitting**: Fit normalizers on training data subset
3. **Feature Generation**: Apply all feature modules with time ordering
4. **Player Aggregation**: Aggregate player features to team level
5. **Feature Merging**: Merge all features by `game_id` with validation
6. **GSI Computation**: Calculate Game Strength Index using fitted normalizers
7. **Final Validation**: Comprehensive validation and cleanup
8. **Output Generation**: Save features with timestamp

## 📊 Feature Output

### Expected Feature Count
- **Total Features**: >30 columns (requirement met)
- **Team Features**: ~15-20 columns
- **Player Features**: ~8-12 columns  
- **Dynamic Features**: ~10-15 columns
- **Market Features**: ~12-18 columns
- **GSI Components**: 5 columns (gsi, gsi_*, gsi_category)

### Data Quality
- **No Missing Values**: All missing values handled with safe defaults
- **No Duplicates**: Automatic duplicate detection and removal
- **Consistent Types**: Proper data types for all features
- **Normalized Ranges**: All numeric features in reasonable ranges

## 🧪 Testing & Validation

### Pipeline Testing
```bash
# Test the feature pipeline
python3 -m src.features.feature_pipeline

# Test feature exploration
python3 notebooks/feature_exploration.py
```

### Validation Checks
- **Schema Integrity**: All merges succeed without silent row dropping
- **Leakage Prevention**: No future data used in feature computation
- **Normalization Safety**: Train/test split prevents data leakage
- **Missing Data Handling**: Graceful degradation with sensible defaults
- **Feature Richness**: >30 features generated consistently

## 🚀 Usage Examples

### Basic Pipeline Usage
```python
from src.features.feature_pipeline import FeaturePipeline

# Initialize pipeline
pipeline = FeaturePipeline()

# Build features with validation
features = pipeline.build_features(games_df, odds_df, players_df)

# Save features
pipeline.save_features(features)
```

### Feature Analysis
```python
# Analyze feature categories
team_features = [col for col in features.columns if col.startswith('team_')]
player_features = [col for col in features.columns if col.startswith('team_') and 'player' in col]

# GSI analysis
gsi_stats = features['gsi'].describe()
gsi_categories = features['gsi_category'].value_counts()
```

## 🔒 Security & Data Integrity

### Data Leakage Prevention
- **Time-based validation**: All market data validated as pre-game
- **Chronological ordering**: Rolling features computed in time order
- **Training isolation**: Normalizers fitted on training data only
- **Future data guards**: Explicit checks prevent future information usage

### Schema Validation
- **Key existence**: All required keys validated before processing
- **Null value checks**: No null values in primary keys
- **Team consistency**: Team names standardized across datasets
- **Merge verification**: Each merge operation validated for success

### Error Handling
- **Graceful degradation**: Pipeline continues with missing data
- **Clear error messages**: Specific error messages for debugging
- **Fallback mechanisms**: Sensible defaults when data unavailable
- **Comprehensive logging**: Detailed logging of all operations

## 📈 Performance & Scalability

### Optimization Features
- **Efficient merges**: All merges use indexed `game_id` columns
- **Vectorized operations**: NumPy/Pandas operations for speed
- **Memory management**: Efficient DataFrame operations and cleanup
- **Batch processing**: Designed for large datasets

### Scalability Considerations
- **Modular design**: Easy to add new feature modules
- **Configurable parameters**: Adjustable thresholds and parameters
- **Memory efficient**: Minimal memory overhead during processing
- **Parallel ready**: Structure supports future parallelization

## 🔮 Future Enhancements

### Phase 3 Integration
- **ML Model Features**: Features designed for ML model consumption
- **Feature Selection**: Built-in feature importance analysis
- **Cross-validation**: Proper train/test splits for production
- **Model Monitoring**: Feature drift detection capabilities

### Advanced Features
- **Real-time Updates**: Support for streaming data updates
- **Feature Store**: Integration with feature store systems
- **A/B Testing**: Support for feature experimentation
- **Performance Monitoring**: Runtime performance metrics

## ✅ Acceptance Criteria Met

- [x] **Schema & Key Integrity**: All joins succeed, no silent row dropping
- [x] **Data Leakage Prevention**: No future data used in features
- [x] **Safe Normalization**: Train/test split prevents leakage
- [x] **Missing Data Handling**: Graceful degradation with defaults
- [x] **Feature Richness**: >30 features generated consistently
- [x] **GSI Formula**: Exact formula implementation with proper normalization
- [x] **Pipeline Execution**: Both pipeline and exploration scripts run successfully
- [x] **Documentation**: Comprehensive documentation of all features and processes

## 🎯 Phase 3 Readiness

Phase 2 is **FULLY COMPLETE AND READY FOR PHASE 3**. The hardened feature engineering pipeline provides:

1. **Production-ready features** with comprehensive validation
2. **Leakage-free feature generation** suitable for ML training
3. **Rich feature set** (>30 features) covering all required categories
4. **Robust error handling** and graceful degradation
5. **Clear documentation** and usage examples
6. **Comprehensive testing** and validation

The system is ready to proceed to Phase 3: ML Model Training with confidence that the feature engineering foundation is solid, secure, and production-ready.