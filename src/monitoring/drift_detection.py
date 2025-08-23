"""
Drift Detection module for Phase 5: Monitoring & CI/CD.
Monitors for data drift between reference (historical) and current (production) datasets.
"""

from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Data class for storing drift detection results for a single feature."""
    feature_name: str
    psi_score: float
    ks_statistic: float
    kl_divergence: float
    drift_detected: bool
    severity: str  # 'none', 'low', 'medium', 'high'
    details: Dict[str, Any]


class DriftDetector:
    """
    Drift detection system for monitoring data distribution changes.
    
    This class implements statistical methods to detect data drift:
    - Population Stability Index (PSI)
    - Kolmogorov-Smirnov (KS) statistic
    - Kullback-Leibler (KL) divergence
    
    Drift is detected when statistical measures exceed predefined thresholds.
    """
    
    def __init__(self, reference_df: pd.DataFrame, 
                 psi_threshold: float = 0.25,
                 ks_threshold: float = 0.1,
                 kl_threshold: float = 0.1):
        """
        Initialize DriftDetector with reference dataset.
        
        Args:
            reference_df: Baseline/reference dataset for comparison
            psi_threshold: PSI threshold for drift detection (default: 0.25)
            ks_threshold: KS statistic threshold (default: 0.1)
            kl_threshold: KL divergence threshold (default: 0.1)
        """
        self.reference_df = reference_df.copy()
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.kl_threshold = kl_threshold
        
        # Store reference statistics for each numeric column
        self.reference_stats = {}
        self._compute_reference_statistics()
        
        logger.info(f"DriftDetector initialized with {len(self.reference_df.columns)} features")
        logger.info(f"Thresholds - PSI: {psi_threshold}, KS: {ks_threshold}, KL: {kl_threshold}")
    
    def _compute_reference_statistics(self):
        """Compute and store reference statistics for each numeric column."""
        numeric_columns = self.reference_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            col_data = self.reference_df[col].dropna()
            if len(col_data) > 0:
                self.reference_stats[col] = {
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'percentiles': col_data.quantile([0.25, 0.5, 0.75]).to_dict()
                }
        
        logger.info(f"Reference statistics computed for {len(self.reference_stats)} numeric features")
    
    def compute_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """
        Compute Population Stability Index (PSI) between reference and current distributions.
        
        PSI measures the change in distribution of a variable between two datasets.
        Higher PSI values indicate greater distribution changes.
        
        Formula: PSI = Σ ( (actual% - expected%) * ln(actual% / expected%) )
        
        Args:
            reference: Reference dataset values
            current: Current dataset values
            bins: Number of bins for histogram calculation
            
        Returns:
            float: PSI score (0 = no drift, higher values indicate drift)
        """
        try:
            # Remove NaN values
            ref_clean = reference[~np.isnan(reference)]
            curr_clean = current[~np.isnan(current)]
            
            if len(ref_clean) == 0 or len(curr_clean) == 0:
                logger.warning("Empty dataset after removing NaN values")
                return np.nan
            
            # Create histogram bins based on reference data
            ref_min, ref_max = ref_clean.min(), ref_clean.max()
            
            # Handle edge case where all values are the same
            if ref_min == ref_max:
                ref_min -= 1e-6
                ref_max += 1e-6
            
            # Create bins and compute histograms
            bin_edges = np.linspace(ref_min, ref_max, bins + 1)
            
            ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
            curr_counts, _ = np.histogram(curr_clean, bins=bin_edges)
            
            # Convert to percentages
            ref_pct = ref_counts / np.sum(ref_counts)
            curr_pct = curr_counts / np.sum(curr_counts)
            
            # Handle zero probabilities (add small epsilon)
            epsilon = 1e-10
            ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
            curr_pct = np.where(curr_pct == 0, epsilon, curr_pct)
            
            # Compute PSI: Σ ( (actual% - expected%) * ln(actual% / expected%) )
            psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
            
            logger.debug(f"PSI computed: {psi:.6f} (bins: {bins})")
            return float(psi)
            
        except Exception as e:
            logger.error(f"Error computing PSI: {e}")
            return np.nan
    
    def compute_ks(self, reference: np.ndarray, current: np.ndarray) -> float:
        """
        Compute Kolmogorov-Smirnov (KS) statistic between reference and current distributions.
        
        KS statistic measures the maximum difference between cumulative distribution functions.
        Higher values indicate greater distribution differences.
        
        Formula: KS = max |CDF_ref(x) - CDF_cur(x)|
        
        Args:
            reference: Reference dataset values
            current: Current dataset values
            
        Returns:
            float: KS statistic (0 = identical distributions, 1 = completely different)
        """
        try:
            # Remove NaN values
            ref_clean = reference[~np.isnan(reference)]
            curr_clean = current[~np.isnan(current)]
            
            if len(ref_clean) == 0 or len(curr_clean) == 0:
                logger.warning("Empty dataset after removing NaN values")
                return np.nan
            
            # Use scipy's KS test
            ks_stat, _ = stats.ks_2samp(ref_clean, curr_clean)
            
            logger.debug(f"KS statistic computed: {ks_stat:.6f}")
            return float(ks_stat)
            
        except Exception as e:
            logger.error(f"Error computing KS statistic: {e}")
            return np.nan
    
    def compute_kl(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """
        Compute Kullback-Leibler (KL) divergence between reference and current distributions.
        
        KL divergence measures the difference between two probability distributions.
        Higher values indicate greater distribution differences.
        
        Formula: KL = Σ p(x) * log(p(x) / q(x))
        
        Args:
            reference: Reference dataset values
            current: Current dataset values
            bins: Number of bins for histogram calculation
            
        Returns:
            float: KL divergence (0 = identical distributions, higher values indicate drift)
        """
        try:
            # Remove NaN values
            ref_clean = reference[~np.isnan(reference)]
            curr_clean = current[~np.isnan(current)]
            
            if len(ref_clean) == 0 or len(curr_clean) == 0:
                logger.warning("Empty dataset after removing NaN values")
                return np.nan
            
            # Create histogram bins based on combined data range
            combined_min = min(ref_clean.min(), curr_clean.min())
            combined_max = max(ref_clean.max(), curr_clean.max())
            
            # Handle edge case where all values are the same
            if combined_min == combined_max:
                combined_min -= 1e-6
                combined_max += 1e-6
            
            # Create bins and compute histograms
            bin_edges = np.linspace(combined_min, combined_max, bins + 1)
            
            ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
            curr_counts, _ = np.histogram(curr_clean, bins=bin_edges)
            
            # Convert to probabilities
            ref_pct = ref_counts / np.sum(ref_counts)
            curr_pct = curr_counts / np.sum(curr_counts)
            
            # Handle zero probabilities (add small epsilon)
            epsilon = 1e-10
            ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
            curr_pct = np.where(curr_pct == 0, epsilon, curr_pct)
            
            # Compute KL divergence: Σ p(x) * log(p(x) / q(x))
            # Note: KL divergence is asymmetric, we compute KL(ref || current)
            kl = np.sum(ref_pct * np.log(ref_pct / curr_pct))
            
            logger.debug(f"KL divergence computed: {kl:.6f} (bins: {bins})")
            return float(kl)
            
        except Exception as e:
            logger.error(f"Error computing KL divergence: {e}")
            return np.nan
    
    def _assess_drift_severity(self, psi: float, ks: float, kl: float) -> Tuple[bool, str]:
        """
        Assess drift severity based on statistical measures.
        
        Args:
            psi: Population Stability Index score
            ks: Kolmogorov-Smirnov statistic
            kl: Kullback-Leibler divergence
            
        Returns:
            Tuple of (drift_detected, severity_level)
        """
        # Check if any measure exceeds thresholds
        psi_drift = psi > self.psi_threshold
        ks_drift = ks > self.ks_threshold
        kl_drift = kl > self.kl_threshold
        
        drift_detected = psi_drift or ks_drift or kl_drift
        
        if not drift_detected:
            return False, "none"
        
        # Determine severity based on number of exceeded thresholds and values
        exceeded_thresholds = sum([psi_drift, ks_drift, kl_drift])
        
        if exceeded_thresholds == 1:
            severity = "low"
        elif exceeded_thresholds == 2:
            severity = "medium"
        else:  # 3 thresholds exceeded
            severity = "high"
        
        # Adjust severity based on magnitude
        if max(psi, ks, kl) > 2 * max(self.psi_threshold, self.ks_threshold, self.kl_threshold):
            severity = "high"
        
        return True, severity
    
    def detect_drift(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect drift for all features between reference and current datasets.
        
        This method runs all drift detection tests (PSI, KS, KL) for each numeric feature
        and returns a comprehensive summary of drift detection results.
        
        Args:
            current_df: Current/production dataset for comparison
            
        Returns:
            Dictionary containing drift detection summary with:
            - overall_drift_detected: Boolean indicating if any drift was detected
            - drift_summary: Summary statistics
            - feature_results: Detailed results for each feature
            - recommendations: Suggested actions based on drift detection
        """
        logger.info("Starting drift detection analysis")
        
        if current_df.empty:
            logger.error("Current DataFrame is empty")
            return {
                'overall_drift_detected': False,
                'drift_summary': {'error': 'Current DataFrame is empty'},
                'feature_results': {},
                'recommendations': ['Provide non-empty current dataset']
            }
        
        # Get numeric columns that exist in both datasets
        ref_numeric = set(self.reference_df.select_dtypes(include=[np.number]).columns)
        curr_numeric = set(current_df.select_dtypes(include=[np.number]).columns)
        common_numeric = ref_numeric.intersection(curr_numeric)
        
        if not common_numeric:
            logger.warning("No common numeric columns found between reference and current datasets")
            return {
                'overall_drift_detected': False,
                'drift_summary': {'warning': 'No common numeric columns'},
                'feature_results': {},
                'recommendations': ['Ensure datasets have compatible numeric columns']
            }
        
        logger.info(f"Analyzing drift for {len(common_numeric)} numeric features")
        
        feature_results = {}
        drift_detected_count = 0
        total_features = len(common_numeric)
        
        for feature in common_numeric:
            logger.debug(f"Analyzing drift for feature: {feature}")
            
            try:
                # Extract feature data
                ref_data = self.reference_df[feature].values
                curr_data = current_df[feature].values
                
                # Compute drift measures
                psi_score = self.compute_psi(ref_data, curr_data)
                ks_statistic = self.compute_ks(ref_data, curr_data)
                kl_divergence = self.compute_kl(ref_data, curr_data)
                
                # Assess drift severity
                drift_detected, severity = self._assess_drift_severity(
                    psi_score, ks_statistic, kl_divergence
                )
                
                if drift_detected:
                    drift_detected_count += 1
                
                # Store results
                feature_results[feature] = DriftResult(
                    feature_name=feature,
                    psi_score=psi_score,
                    ks_statistic=ks_statistic,
                    kl_divergence=kl_divergence,
                    drift_detected=drift_detected,
                    severity=severity,
                    details={
                        'reference_count': len(ref_data),
                        'current_count': len(curr_data),
                        'reference_mean': np.nanmean(ref_data),
                        'current_mean': np.nanmean(curr_data),
                        'reference_std': np.nanstd(ref_data),
                        'current_std': np.nanstd(curr_data)
                    }
                )
                
                logger.debug(f"Feature {feature}: drift={drift_detected}, severity={severity}")
                
            except Exception as e:
                logger.error(f"Error analyzing drift for feature {feature}: {e}")
                feature_results[feature] = DriftResult(
                    feature_name=feature,
                    psi_score=np.nan,
                    ks_statistic=np.nan,
                    kl_divergence=np.nan,
                    drift_detected=False,
                    severity="error",
                    details={'error': str(e)}
                )
        
        # Generate drift summary
        overall_drift_detected = drift_detected_count > 0
        drift_percentage = (drift_detected_count / total_features) * 100 if total_features > 0 else 0
        
        drift_summary = {
            'total_features_analyzed': total_features,
            'features_with_drift': drift_detected_count,
            'drift_percentage': drift_percentage,
            'psi_threshold': self.psi_threshold,
            'ks_threshold': self.ks_threshold,
            'kl_threshold': self.kl_threshold
        }
        
        # Generate recommendations
        recommendations = []
        if overall_drift_detected:
            if drift_percentage > 50:
                recommendations.append("High drift detected - consider retraining model")
                recommendations.append("Investigate data pipeline for systematic changes")
            elif drift_percentage > 25:
                recommendations.append("Moderate drift detected - monitor closely")
                recommendations.append("Consider feature importance analysis")
            else:
                recommendations.append("Low drift detected - continue monitoring")
        else:
            recommendations.append("No drift detected - system is stable")
        
        recommendations.append("Review drift thresholds if false positives occur")
        
        # Compile final results
        results = {
            'overall_drift_detected': overall_drift_detected,
            'drift_summary': drift_summary,
            'feature_results': feature_results,
            'recommendations': recommendations
        }
        
        if overall_drift_detected:
            logger.warning(f"Drift detected in {drift_detected_count}/{total_features} features ({drift_percentage:.1f}%)")
        else:
            logger.info(f"No drift detected across {total_features} features")
        
        return results
    
    def get_drift_report(self, drift_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable drift detection report.
        
        Args:
            drift_results: Results from detect_drift method
            
        Returns:
            String containing formatted drift detection report
        """
        report = []
        report.append("=" * 70)
        report.append("DRIFT DETECTION REPORT")
        report.append("=" * 70)
        
        # Overall status
        status = "🚨 DRIFT DETECTED" if drift_results['overall_drift_detected'] else "✅ NO DRIFT DETECTED"
        report.append(f"Overall Status: {status}")
        report.append("")
        
        # Summary statistics
        summary = drift_results['drift_summary']
        report.append("Summary Statistics:")
        report.append(f"  Total Features Analyzed: {summary['total_features_analyzed']}")
        report.append(f"  Features with Drift: {summary['features_with_drift']}")
        report.append(f"  Drift Percentage: {summary['drift_percentage']:.1f}%")
        report.append(f"  PSI Threshold: {summary['psi_threshold']}")
        report.append(f"  KS Threshold: {summary['ks_threshold']}")
        report.append(f"  KL Threshold: {summary['kl_threshold']}")
        report.append("")
        
        # Feature-level results
        if drift_results['feature_results']:
            report.append("Feature-Level Results:")
            report.append("-" * 50)
            
            for feature_name, result in drift_results['feature_results'].items():
                drift_icon = "🚨" if result.drift_detected else "✅"
                report.append(f"{drift_icon} {feature_name}:")
                report.append(f"    PSI: {result.psi_score:.6f}")
                report.append(f"    KS: {result.ks_statistic:.6f}")
                report.append(f"    KL: {result.kl_divergence:.6f}")
                report.append(f"    Severity: {result.severity.upper()}")
                report.append("")
        
        # Recommendations
        if drift_results['recommendations']:
            report.append("Recommendations:")
            report.append("-" * 20)
            for i, rec in enumerate(drift_results['recommendations'], 1):
                report.append(f"  {i}. {rec}")
            report.append("")
        
        report.append("=" * 70)
        
        return "\n".join(report)


# Example usage and testing
if __name__ == "__main__":
    # Create sample reference and current data for testing
    np.random.seed(42)
    
    # Reference data (baseline)
    reference_data = {
        'feature_1': np.random.normal(0, 1, 1000),
        'feature_2': np.random.uniform(0, 1, 1000),
        'feature_3': np.random.exponential(1, 1000)
    }
    reference_df = pd.DataFrame(reference_data)
    
    # Current data (with some drift)
    current_data = {
        'feature_1': np.random.normal(0.2, 1.1, 1000),  # Slight drift
        'feature_2': np.random.uniform(0, 1, 1000),      # No drift
        'feature_3': np.random.exponential(1.5, 1000)    # Moderate drift
    }
    current_df = pd.DataFrame(current_data)
    
    print("Testing Drift Detection Module...")
    print()
    
    # Test drift detection
    detector = DriftDetector(reference_df)
    results = detector.detect_drift(current_df)
    
    # Print report
    report = detector.get_drift_report(results)
    print(report)