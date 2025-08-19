# Jupyter Notebooks for CBB Betting ML System

This directory contains Jupyter notebooks for data exploration, analysis, and model development.

## Available Notebooks

### 1. Data Exploration (`01_data_exploration.ipynb`)
- **Purpose**: Explore collected data quality and structure
- **Features**: Data loading, quality analysis, basic visualizations
- **Prerequisites**: Raw data files in `../data/raw/`

### 2. Feature Analysis (Coming Soon)
- **Purpose**: Analyze feature distributions and relationships
- **Features**: Statistical analysis, correlation matrices, feature importance
- **Prerequisites**: Processed data from ETL pipeline

### 3. Model Development (Coming Soon)
- **Purpose**: Develop and evaluate ML models
- **Features**: Model training, validation, performance analysis
- **Prerequisites**: Feature-engineered datasets

## Getting Started

1. **Install Jupyter**:
   ```bash
   pip install jupyter
   ```

2. **Start Jupyter Server**:
   ```bash
   cd cbb_model/notebooks
   jupyter notebook
   ```

3. **Open Notebooks**: Navigate to the desired notebook file

## Data Dependencies

- **Raw Data**: Located in `../data/raw/`
- **Processed Data**: Located in `../data/processed/`
- **Configuration**: Located in `../config.yaml`

## Tips for Development

- Start with data exploration to understand data quality
- Use small datasets for testing before running on full data
- Document findings and insights in notebook markdown cells
- Save intermediate results to avoid re-running expensive operations

## Example Workflow

1. Run data collection scripts to populate `../data/raw/`
2. Use `01_data_exploration.ipynb` to assess data quality
3. Run ETL pipeline to create processed datasets
4. Develop features and models in subsequent notebooks
5. Document insights and next steps

## Contributing

When adding new notebooks:
- Use descriptive filenames with numbering
- Include clear markdown documentation
- Test with sample data before committing
- Update this README with new notebook descriptions