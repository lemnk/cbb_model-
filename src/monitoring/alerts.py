"""
Alerts module for Phase 5: Monitoring & CI/CD.
Triggers alerts when monitoring checks fail in the CBB Betting ML System.
"""

from typing import Dict, List, Any, Optional
import os
import requests
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertManager:
    """
    Alert management system for monitoring failures.
    
    This class handles alert generation and delivery through multiple channels:
    - Console output for development/testing
    - File logging for persistent storage
    - Slack webhook for team notifications
    """
    
    def __init__(self, alert_config: Dict[str, Any]):
        """
        Initialize AlertManager with configuration.
        
        Args:
            alert_config: Dictionary containing alert configuration from alerts.yml
        """
        self.config = alert_config.copy()
        self.mode = self.config.get('mode', 'console')
        self.slack_webhook = self.config.get('slack_webhook', '')
        self.file_path = self.config.get('file_path', 'logs/alerts.log')
        
        # Validate configuration
        self._validate_config()
        
        # Initialize file path if needed
        if self.mode == 'file':
            self._ensure_log_directory()
        
        logger.info(f"AlertManager initialized with mode: {self.mode}")
    
    def _validate_config(self):
        """Validate alert configuration."""
        valid_modes = ['console', 'file', 'slack']
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid alert mode: {self.mode}. Must be one of {valid_modes}")
        
        if self.mode == 'slack':
            if not self.slack_webhook or self.slack_webhook == "https://hooks.slack.com/services/XXXX/XXXX/XXXX":
                logger.warning("Slack webhook not configured properly")
        
        if self.mode == 'file':
            if not self.file_path:
                raise ValueError("File path must be specified for file mode")
    
    def _ensure_log_directory(self):
        """Ensure log directory exists for file mode."""
        log_dir = Path(self.file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured log directory exists: {log_dir}")
    
    def check_alerts(self, results: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Scan monitoring results for alerts.
        
        Args:
            results: Dictionary of results from monitoring modules
                    Format: {metric_name: {"value": float, "threshold": float, "status": str}}
        
        Returns:
            List of alert messages for metrics with "ALERT" status
        """
        if not results:
            logger.info("No results provided for alert checking")
            return []
        
        alert_messages = []
        
        for metric_name, metric_result in results.items():
            if not isinstance(metric_result, dict):
                logger.warning(f"Invalid result format for metric {metric_name}")
                continue
            
            status = metric_result.get('status', '')
            
            if status == 'ALERT':
                message = self._generate_alert_message(metric_name, metric_result)
                alert_messages.append(message)
                logger.debug(f"Alert generated for {metric_name}: {message}")
        
        logger.info(f"Generated {len(alert_messages)} alerts from {len(results)} metrics")
        return alert_messages
    
    def _generate_alert_message(self, metric_name: str, metric_result: Dict[str, Any]) -> str:
        """
        Generate alert message for a specific metric.
        
        Args:
            metric_name: Name of the metric
            metric_result: Dictionary containing metric value, threshold, and status
            
        Returns:
            Formatted alert message
        """
        value = metric_result.get('value', 'unknown')
        threshold = metric_result.get('threshold', 'unknown')
        
        # Determine if this is a "higher is better" or "lower is better" metric
        if metric_name in ['log_loss', 'brier_score']:
            # For these metrics, lower is better, so alert when value > threshold
            comparison = "above"
            symbol = ">"
        else:
            # For most metrics, higher is better, so alert when value < threshold
            comparison = "below"
            symbol = "<"
        
        # Format the message
        if isinstance(value, (int, float)) and isinstance(threshold, (int, float)):
            message = f"[ALERT] {metric_name} {comparison} threshold: {value:.6f} {symbol} {threshold:.6f}"
        else:
            message = f"[ALERT] {metric_name} failed threshold check: {value} vs {threshold}"
        
        return message
    
    def send_alerts(self, messages: List[str]):
        """
        Send alerts through the configured delivery method.
        
        Args:
            messages: List of alert messages to send
        """
        if not messages:
            logger.info("No alert messages to send")
            return
        
        logger.info(f"Sending {len(messages)} alerts via {self.mode} mode")
        
        try:
            if self.mode == 'console':
                self._send_console_alerts(messages)
            elif self.mode == 'file':
                self._send_file_alerts(messages)
            elif self.mode == 'slack':
                self._send_slack_alerts(messages)
            else:
                logger.error(f"Unknown alert mode: {self.mode}")
        except Exception as e:
            logger.error(f"Failed to send alerts: {e}")
            # Fallback to console if other methods fail
            if self.mode != 'console':
                logger.info("Falling back to console output")
                self._send_console_alerts(messages)
    
    def _send_console_alerts(self, messages: List[str]):
        """Send alerts to console/stdout."""
        logger.info("Sending alerts to console")
        for message in messages:
            print(message)
    
    def _send_file_alerts(self, messages: List[str]):
        """Send alerts to log file."""
        logger.info(f"Sending alerts to file: {self.file_path}")
        
        try:
            with open(self.file_path, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().isoformat()
                for message in messages:
                    f.write(f"{timestamp} - {message}\n")
                f.flush()
            
            logger.info(f"Successfully wrote {len(messages)} alerts to {self.file_path}")
        except Exception as e:
            logger.error(f"Failed to write alerts to file: {e}")
            raise
    
    def _send_slack_alerts(self, messages: List[str]):
        """Send alerts to Slack via webhook."""
        if not self.slack_webhook or self.slack_webhook == "https://hooks.slack.com/services/XXXX/XXXX/XXXX":
            logger.error("Slack webhook not properly configured")
            raise ValueError("Invalid Slack webhook URL")
        
        logger.info(f"Sending alerts to Slack webhook")
        
        # Format messages for Slack
        slack_text = "🚨 *CBB Betting ML System Alerts* 🚨\n\n"
        for message in messages:
            # Remove [ALERT] prefix for cleaner Slack formatting
            clean_message = message.replace("[ALERT] ", "• ")
            slack_text += f"{clean_message}\n"
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        slack_text += f"\n_Generated at {timestamp}_"
        
        payload = {
            "text": slack_text,
            "username": "CBB-ML-Monitor",
            "icon_emoji": ":warning:"
        }
        
        try:
            response = requests.post(
                self.slack_webhook,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            logger.info(f"Successfully sent {len(messages)} alerts to Slack")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send alerts to Slack: {e}")
            raise
    
    def get_alert_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of alert status.
        
        Args:
            results: Dictionary of monitoring results
            
        Returns:
            Dictionary containing alert summary statistics
        """
        if not results:
            return {
                'total_metrics': 0,
                'alerts': 0,
                'warnings': 0,
                'passes': 0,
                'alert_rate': 0.0
            }
        
        status_counts = {'ALERT': 0, 'WARNING': 0, 'PASS': 0}
        
        for metric_result in results.values():
            if isinstance(metric_result, dict):
                status = metric_result.get('status', 'UNKNOWN')
                if status in status_counts:
                    status_counts[status] += 1
        
        total_metrics = len(results)
        alert_rate = (status_counts['ALERT'] / total_metrics * 100) if total_metrics > 0 else 0.0
        
        return {
            'total_metrics': total_metrics,
            'alerts': status_counts['ALERT'],
            'warnings': status_counts['WARNING'],
            'passes': status_counts['PASS'],
            'alert_rate': round(alert_rate, 2)
        }
    
    def format_alert_summary(self, summary: Dict[str, Any]) -> str:
        """
        Format alert summary as human-readable string.
        
        Args:
            summary: Alert summary from get_alert_summary()
            
        Returns:
            Formatted summary string
        """
        if summary['total_metrics'] == 0:
            return "No metrics to summarize"
        
        lines = [
            "Alert Summary:",
            f"  Total Metrics: {summary['total_metrics']}",
            f"  🚨 Alerts: {summary['alerts']}",
            f"  ⚠️ Warnings: {summary['warnings']}",
            f"  ✅ Passes: {summary['passes']}",
            f"  Alert Rate: {summary['alert_rate']}%"
        ]
        
        return "\n".join(lines)


# Example usage and testing
if __name__ == "__main__":
    # Sample alert configuration
    sample_config = {
        'mode': 'console',
        'slack_webhook': 'https://hooks.slack.com/services/XXXX/XXXX/XXXX',
        'file_path': 'logs/alerts.log'
    }
    
    print("Testing Alert Manager...")
    print()
    
    # Sample monitoring results with alerts
    sample_results = {
        "accuracy": {"value": 0.52, "threshold": 0.55, "status": "ALERT"},
        "log_loss": {"value": 0.82, "threshold": 0.7, "status": "ALERT"},
        "precision": {"value": 0.48, "threshold": 0.5, "status": "WARNING"},
        "recall": {"value": 0.55, "threshold": 0.5, "status": "PASS"},
        "expected_value": {"value": -0.03, "threshold": 0.0, "status": "ALERT"}
    }
    
    # Test alert manager
    manager = AlertManager(sample_config)
    
    # Check for alerts
    alert_messages = manager.check_alerts(sample_results)
    print(f"Generated {len(alert_messages)} alert messages:")
    for message in alert_messages:
        print(f"  {message}")
    
    print()
    
    # Send alerts
    print("Sending alerts...")
    manager.send_alerts(alert_messages)
    
    print()
    
    # Generate summary
    summary = manager.get_alert_summary(sample_results)
    summary_text = manager.format_alert_summary(summary)
    print(summary_text)