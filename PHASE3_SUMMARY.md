# Phase 3: Model Training & Evaluation - CBB Betting ML System

## 🎯 Overview

Phase 3 implements and evaluates machine learning models to predict game outcomes (classification) and point differentials (regression) using the hardened feature pipeline from Phase 2. All formulas, evaluation metrics, and ROI simulation are explicitly implemented as specified in the requirements.

## 📊 Models Trained

### 1. Logistic Regression (Classification Only)
- **Purpose**: Game outcome prediction (win/loss)
- **Implementation**: `src/models/logistic_regression.py`
- **Features**: Binary classification with probability outputs
- **Hyperparameters**: L2 regularization, max_iter=1000

### 2. Random Forest (Classification + Regression)
- **Purpose**: Game outcome prediction and point differential prediction
- **Implementation**: `src/models/random_forest.py`
- **Features**: Ensemble method with feature importance
- **Hyperparameters**: n_estimators=100, max_depth=10, min_samples_split=5

### 3. XGBoost (Classification + Regression)
- **Purpose**: Game outcome prediction and point differential prediction
- **Implementation**: `src/models/xgboost_model.py`
- **Features**: Gradient boosting with early stopping
- **Hyperparameters**: n_estimators=100, max_depth=6, learning_rate=0.1

### 4. Neural Network (Classification + Regression)
- **Purpose**: Game outcome prediction and point differential prediction
- **Implementation**: `src/models/neural_network.py`
- **Features**: PyTorch-based with dropout and batch normalization
- **Architecture**: 128→64→32 hidden layers with ReLU activation

## 🧮 Formulas (Exact Implementation)

### Classification Metrics

#### Accuracy
```
accuracy = (tp + tn) / (tp + tn + fp + fn)
```
**Implementation**: `evaluation.py` line 67

#### Precision
```
precision = tp / (tp + fp)
```
**Implementation**: `evaluation.py` line 78

#### Recall
```
recall = tp / (tp + fn)
```
**Implementation**: `evaluation.py` line 89

#### F1-Score
```
f1_score = 2 * (precision * recall) / (precision + recall)
```
**Implementation**: `evaluation.py` line 100

#### AUC
```
auc = roc_auc_score(y_true, y_pred_proba)
```
**Implementation**: `evaluation.py` line 158 (using sklearn as specified)

### Regression Metrics

#### RMSE
```
rmse = sqrt(mean((y_true - y_pred)²))
```
**Implementation**: `evaluation.py` line 112

#### MAE
```
mae = mean(|y_true - y_pred|)
```
**Implementation**: `evaluation.py` line 123

#### R²
```
r2 = 1 - (ss_res / ss_tot)
```
Where:
- `ss_res = sum((y_true - y_pred)²)`
- `ss_tot = sum((y_true - mean(y_true))²)`

**Implementation**: `evaluation.py` line 134

### ROI & Kelly Simulation

#### Edge
```
Edge = p̂ ⋅ (O - 1) - (1 - p̂)
```
Where:
- `p̂` = Model predicted probability
- `O` = Decimal odds

**Implementation**: `roi_simulator.py` line 47

#### ROI
```
ROI = Total Profit / Total Stakes
```
**Implementation**: `roi_simulator.py` line 58

#### Kelly Fraction
```
f* = (bp - q) / b
```
Where:
- `b = O - 1`
- `p = p̂` (model prediction)
- `q = 1 - p`

**Implementation**: `roi_simulator.py` line 70

## 🚀 Train/Val/Test Procedure

### Data Split Implementation
**Exact code from requirements**:

```python
# First split: 70% train, 30% temp
train_df, temp_df = train_test_split(
    df, 
    test_size=0.3, 
    stratify=df['season'], 
    random_state=42
)

# Second split: 15% val, 15% test (50% of temp)
val_df, test_df = train_test_split(
    temp_df, 
    test_size=0.5, 
    stratify=temp_df['season'], 
    random_state=42
)
```

**Implementation**: `train_models.py` lines 218-232

### Final Split Distribution
- **Training Set**: 70% (as specified in requirements)
- **Validation Set**: 15% (as specified in requirements)  
- **Test Set**: 15% (as specified in requirements)
- **Stratification**: By season to maintain distribution

## 📈 Metrics Results

### Classification Performance
All models are evaluated using the exact formulas above:

- **Accuracy**: Overall correctness
- **Precision**: True positive rate among positive predictions
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve (using sklearn as specified)

### Regression Performance
All models are evaluated using the exact formulas above:

- **RMSE**: Root mean squared error
- **MAE**: Mean absolute error
- **R²**: Coefficient of determination

### Model Comparison
The system trains and compares **3+ models** as required:
1. Logistic Regression (classification only)
2. Random Forest (classification + regression)
3. XGBoost (classification + regression)
4. Neural Network (classification + regression)

## 💰 ROI Simulation Findings

### Betting Strategies Implemented
1. **Flat Betting**: Fixed bet size (2% of bankroll)
2. **Kelly Betting**: Optimal bet size based on edge
3. **Edge-Based Betting**: Only bet when edge > 0

### Key Metrics
- **Edge Calculation**: Using exact formula from requirements
- **ROI Analysis**: Profit/loss tracking over time
- **Risk Management**: Kelly fraction capping at 10%
- **Bankroll Evolution**: Cumulative performance tracking

### Visualization Outputs
- **Bankroll Evolution**: Strategy comparison over time
- **Edge vs Profit**: Scatter plots for each strategy
- **Performance Metrics**: Comprehensive strategy comparison

## 🔧 Technical Implementation

### File Structure
```
src/models/
├── __init__.py
├── logistic_regression.py
├── random_forest.py
├── xgboost_model.py
└── neural_network.py

train_models.py          # Main training pipeline
evaluation.py            # Evaluation metrics & visualization
roi_simulator.py         # ROI & Kelly calculations
tests/
└── test_evaluation.py   # Unit tests for formulas
outputs/phase3/
├── models/              # Trained model files
└── plots/               # Visualization outputs
```

### Key Features
- **Reproducibility**: Fixed random seed (42) throughout
- **Modular Design**: Each model is a separate class
- **Comprehensive Evaluation**: All metrics using exact formulas
- **Hyperparameter Tuning**: Grid search on validation set
- **Model Persistence**: Save/load functionality with joblib
- **Visualization**: Confusion matrices, ROC curves, feature importance

### Dependencies
- **Core ML**: scikit-learn, XGBoost, PyTorch
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Utilities**: joblib, os, datetime

## 📊 Feature Output

### Classification Features
- **Target**: Binary win/loss (0/1)
- **Input**: All engineered features from Phase 2
- **Output**: Probability predictions and binary classifications

### Regression Features
- **Target**: Point differential (team_score - opponent_score)
- **Input**: All engineered features from Phase 2
- **Output**: Continuous point differential predictions

### Feature Engineering Integration
- **Phase 2 Pipeline**: Seamless integration with hardened features
- **Normalization**: StandardScaler applied consistently
- **Feature Selection**: Automatic exclusion of non-feature columns
- **Validation**: Comprehensive input validation and error handling

## 🧪 Testing & Validation

### Unit Tests
**File**: `tests/test_evaluation.py`

**Coverage**:
- All classification metric formulas
- All regression metric formulas
- ROI simulator formulas
- Edge cases and boundary conditions
- Random state consistency

**Test Execution**:
```bash
python -m pytest tests/test_evaluation.py -v
```

### Formula Verification
Every metric implementation is tested against the exact mathematical formulas:
- **Accuracy**: Verified with confusion matrix components
- **Precision/Recall**: Verified with true/false positive counts
- **F1-Score**: Verified with harmonic mean calculation
- **RMSE/MAE/R²**: Verified with numpy calculations
- **Edge/ROI/Kelly**: Verified with mathematical formulas

## 🎯 Phase 4 Readiness

### Model Performance Insights
- **Best Performing Models**: Identified through comprehensive evaluation
- **Feature Importance**: Ranked features for interpretability
- **Hyperparameter Optimization**: Best parameters for each model type
- **Performance Baselines**: Established metrics for future comparison

### Production Deployment
- **Model Serialization**: All models saved in joblib format
- **API Ready**: Clean interfaces for prediction and evaluation
- **Scalability**: Efficient feature processing and prediction
- **Monitoring**: Comprehensive evaluation metrics for model tracking

### Next Phase Recommendations
1. **Model Ensemble**: Combine best performing models
2. **Real-time Prediction**: Deploy models for live betting
3. **Performance Monitoring**: Track model drift over time
4. **Feature Evolution**: Iterate on feature engineering based on model insights

## ✅ Requirements Compliance

### Mandatory Requirements Met
- ✅ **3+ Models**: 4 models implemented and compared
- ✅ **Exact Formulas**: All mathematical formulas implemented exactly as specified
- ✅ **Train/Val/Test**: 70%/15%/15% split with stratification
- ✅ **Classification + Regression**: Both tasks supported
- ✅ **Unit Tests**: Comprehensive testing of all formulas
- ✅ **ROI Simulation**: Flat vs Kelly betting strategies
- ✅ **Documentation**: Complete PHASE3_SUMMARY.md

### Formula Accuracy Verification
- ✅ **Classification Metrics**: Accuracy, Precision, Recall, F1-Score, AUC
- ✅ **Regression Metrics**: RMSE, MAE, R²
- ✅ **ROI Formulas**: Edge, ROI, Kelly calculations
- ✅ **Data Split**: Exact train_test_split implementation

### Output Deliverables
- ✅ **Trained Models**: Saved in `outputs/phase3/models/`
- ✅ **Visualizations**: Plots saved in `outputs/phase3/plots/`
- ✅ **Evaluation Reports**: Comprehensive performance analysis
- ✅ **ROI Analysis**: Betting strategy comparison
- ✅ **Documentation**: Complete implementation summary

## 🎉 Conclusion

Phase 3 successfully implements a comprehensive machine learning training and evaluation system for the CBB Betting ML System. All requirements have been met with exact formula implementations, comprehensive model training, and thorough evaluation. The system is now ready for Phase 4 deployment and production use.

**Key Achievements**:
- 4 ML models trained and evaluated
- All mathematical formulas implemented exactly as specified
- Comprehensive ROI simulation with multiple betting strategies
- Production-ready model persistence and evaluation
- Complete testing and documentation

**Next Steps**: Phase 4 - Model Deployment and Live Prediction System