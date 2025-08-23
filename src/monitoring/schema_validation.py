"""
Schema validation module for Phase 5: Monitoring & CI/CD.
Ensures input game data matches the expected structure for the CBB Betting ML System.
"""

from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
from pydantic import BaseModel, validator
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GameRecord(BaseModel):
    """
    Pydantic model for validating individual game records.
    
    Attributes:
        game_id: Unique identifier for the game
        date: Game date in YYYY-MM-DD format
        season: NCAA basketball season year
        home_team: Name of the home team
        away_team: Name of the away team
        team_efficiency: Team efficiency score (0.0 to 1.0)
        player_availability: Player availability score (0.0 to 1.0)
        dynamic_factors: Dynamic factors score (0.0 to 1.0)
        market_signals: Market signals score (0.0 to 1.0)
        target: Binary outcome (0 for loss, 1 for win)
    """
    
    game_id: str
    date: str  # YYYY-MM-DD
    season: int
    home_team: str
    away_team: str
    team_efficiency: float
    player_availability: float
    dynamic_factors: float
    market_signals: float
    target: int  # binary outcome 0/1
    
    @validator('date')
    def validate_date_format(cls, v):
        """Validate date format is YYYY-MM-DD."""
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')
    
    @validator('season')
    def validate_season(cls, v):
        """Validate season is a reasonable year."""
        if v < 2000 or v > 2030:
            raise ValueError('Season must be between 2000 and 2030')
        return v
    
    @validator('team_efficiency', 'player_availability', 'dynamic_factors', 'market_signals')
    def validate_score_range(cls, v):
        """Validate score fields are between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Score fields must be between 0.0 and 1.0')
        return v
    
    @validator('target')
    def validate_target(cls, v):
        """Validate target is binary (0 or 1)."""
        if v not in [0, 1]:
            raise ValueError('Target must be 0 or 1')
        return v


class SchemaValidator:
    """
    Schema validation system for ensuring input game data matches expected structure.
    
    This class provides comprehensive validation for:
    - Required column presence
    - Data type consistency
    - Missing value detection
    - Individual record validation
    """
    
    def __init__(self):
        """Initialize the SchemaValidator with expected schema."""
        # Define required columns and their expected types
        self.required_columns = {
            'game_id': str,
            'date': str,
            'season': int,
            'home_team': str,
            'away_team': str,
            'team_efficiency': float,
            'player_availability': float,
            'dynamic_factors': float,
            'market_signals': float,
            'target': int
        }
        
        logger.info(f"SchemaValidator initialized with {len(self.required_columns)} required columns")
    
    def validate_row(self, row: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate a single row/record against the expected schema.
        
        Args:
            row: Dictionary containing a single game record
            
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: Boolean indicating if validation passed
            - error_message: String describing the error, or None if valid
        """
        try:
            # Validate using Pydantic model
            GameRecord(**row)
            return True, None
        except Exception as e:
            error_msg = f"Row validation failed: {str(e)}"
            logger.warning(f"Row validation failed for game_id {row.get('game_id', 'unknown')}: {str(e)}")
            return False, error_msg
    
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate all rows in a DataFrame against the expected schema.
        
        Args:
            df: Pandas DataFrame containing game records
            
        Returns:
            Tuple of (is_valid, error_messages)
            - is_valid: Boolean indicating if all rows passed validation
            - error_messages: List of error messages for failed rows
        """
        if df.empty:
            return False, ["DataFrame is empty"]
        
        error_messages = []
        valid_rows = 0
        total_rows = len(df)
        
        logger.info(f"Starting validation of {total_rows} rows")
        
        for idx, row in df.iterrows():
            is_valid, error_msg = self.validate_row(row.to_dict())
            if not is_valid:
                error_messages.append(f"Row {idx}: {error_msg}")
            else:
                valid_rows += 1
        
        is_valid = len(error_messages) == 0
        
        logger.info(f"Validation complete: {valid_rows}/{total_rows} rows valid")
        if not is_valid:
            logger.warning(f"Validation failed: {len(error_messages)} errors found")
        
        return is_valid, error_messages
    
    def comprehensive_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive validation including schema, types, and missing values.
        
        Args:
            df: Pandas DataFrame containing game records
            
        Returns:
            Dictionary containing validation summary with:
            - is_valid: Overall validation status
            - schema_valid: Schema validation status
            - type_valid: Data type validation status
            - missing_valid: Missing value validation status
            - error_summary: Summary of all errors found
            - validation_details: Detailed validation results
        """
        logger.info("Starting comprehensive validation")
        
        validation_results = {
            'is_valid': False,
            'schema_valid': False,
            'type_valid': False,
            'missing_valid': False,
            'error_summary': [],
            'validation_details': {}
        }
        
        # 1. Check required columns exist
        missing_columns = [col for col in self.required_columns.keys() if col not in df.columns]
        if missing_columns:
            validation_results['error_summary'].append(f"Missing required columns: {missing_columns}")
            validation_results['schema_valid'] = False
        else:
            validation_results['schema_valid'] = True
            logger.info("All required columns present")
        
        # 2. Check data types
        type_errors = []
        for col, expected_type in self.required_columns.items():
            if col in df.columns:
                # Check if all values in column match expected type
                try:
                    if expected_type == str:
                        # For strings, check if all values are strings
                        if not all(isinstance(val, str) for val in df[col].dropna()):
                            type_errors.append(f"Column '{col}': Expected {expected_type}, found mixed types")
                    elif expected_type == int:
                        # For ints, check if all values are integers
                        if not all(isinstance(val, (int, float)) and val.is_integer() if isinstance(val, float) else True 
                                 for val in df[col].dropna()):
                            type_errors.append(f"Column '{col}': Expected {expected_type}, found non-integer values")
                    elif expected_type == float:
                        # For floats, check if all values are numeric
                        if not pd.api.types.is_numeric_dtype(df[col]):
                            type_errors.append(f"Column '{col}': Expected {expected_type}, found non-numeric values")
                except Exception as e:
                    type_errors.append(f"Column '{col}': Type validation error - {str(e)}")
        
        if type_errors:
            validation_results['error_summary'].extend(type_errors)
            validation_results['type_valid'] = False
        else:
            validation_results['type_valid'] = True
            logger.info("All data types match expected schema")
        
        # 3. Check for missing values
        missing_values = df.isnull().sum()
        columns_with_missing = missing_values[missing_values > 0]
        
        if not columns_with_missing.empty:
            missing_info = [f"Column '{col}': {count} missing values" 
                           for col, count in columns_with_missing.items()]
            validation_results['error_summary'].extend(missing_info)
            validation_results['missing_valid'] = False
            logger.warning(f"Missing values found in columns: {list(columns_with_missing.index)}")
        else:
            validation_results['missing_valid'] = True
            logger.info("No missing values found")
        
        # 4. Validate individual rows using Pydantic
        if validation_results['schema_valid'] and validation_results['type_valid'] and validation_results['missing_valid']:
            schema_valid, schema_errors = self.validate_dataframe(df)
            if not schema_valid:
                validation_results['error_summary'].extend(schema_errors)
                validation_results['schema_valid'] = False
        
        # 5. Determine overall validation status
        validation_results['is_valid'] = (
            validation_results['schema_valid'] and 
            validation_results['type_valid'] and 
            validation_results['missing_valid']
        )
        
        # 6. Add validation details
        validation_results['validation_details'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_columns': missing_columns,
            'columns_with_missing_values': list(columns_with_missing.index) if not columns_with_missing.empty else [],
            'type_errors': type_errors,
            'total_errors': len(validation_results['error_summary'])
        }
        
        if validation_results['is_valid']:
            logger.info("Comprehensive validation PASSED")
        else:
            logger.error(f"Comprehensive validation FAILED: {len(validation_results['error_summary'])} errors")
        
        return validation_results
    
    def get_validation_summary(self, validation_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable validation summary.
        
        Args:
            validation_results: Results from comprehensive_validation method
            
        Returns:
            String containing formatted validation summary
        """
        summary = []
        summary.append("=" * 60)
        summary.append("SCHEMA VALIDATION SUMMARY")
        summary.append("=" * 60)
        
        # Overall status
        status = "✅ PASSED" if validation_results['is_valid'] else "❌ FAILED"
        summary.append(f"Overall Status: {status}")
        summary.append("")
        
        # Individual validation results
        summary.append("Validation Components:")
        summary.append(f"  Schema Validation: {'✅ PASSED' if validation_results['schema_valid'] else '❌ FAILED'}")
        summary.append(f"  Type Validation: {'✅ PASSED' if validation_results['type_valid'] else '❌ FAILED'}")
        summary.append(f"  Missing Values: {'✅ PASSED' if validation_results['missing_valid'] else '❌ FAILED'}")
        summary.append("")
        
        # Error details
        if validation_results['error_summary']:
            summary.append(f"Errors Found ({len(validation_results['error_summary'])}):")
            for i, error in enumerate(validation_results['error_summary'], 1):
                summary.append(f"  {i}. {error}")
        else:
            summary.append("✅ No errors found")
        
        summary.append("")
        
        # Statistics
        details = validation_results['validation_details']
        summary.append("Statistics:")
        summary.append(f"  Total Rows: {details['total_rows']}")
        summary.append(f"  Total Columns: {details['total_columns']}")
        summary.append(f"  Missing Columns: {len(details['missing_columns'])}")
        summary.append(f"  Columns with Missing Values: {len(details['columns_with_missing_values'])}")
        
        summary.append("=" * 60)
        
        return "\n".join(summary)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    sample_data = {
        'game_id': ['G001', 'G002'],
        'date': ['2024-01-15', '2024-01-16'],
        'season': [2024, 2024],
        'home_team': ['Team A', 'Team B'],
        'away_team': ['Team C', 'Team D'],
        'team_efficiency': [0.75, 0.82],
        'player_availability': [0.90, 0.85],
        'dynamic_factors': [0.60, 0.70],
        'market_signals': [0.45, 0.55],
        'target': [1, 0]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Test validation
    validator = SchemaValidator()
    
    print("Testing Schema Validation Module...")
    print()
    
    # Test comprehensive validation
    results = validator.comprehensive_validation(df)
    
    # Print summary
    summary = validator.get_validation_summary(results)
    print(summary)