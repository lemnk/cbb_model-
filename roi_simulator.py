import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os

class ROISimulator:
    """
    ROI Simulator for CBB betting analysis.
    
    Implements exact formulas from requirements:
    - Edge = p̂ ⋅ (O - 1) - (1 - p̂)
    - ROI = Total Profit / Total Stakes
    - Kelly = (bp - q) / b, where b = O - 1, p = p̂, q = 1 - p
    """
    
    def __init__(self, initial_bankroll=10000):
        """
        Initialize ROI Simulator.
        
        Args:
            initial_bankroll: Starting bankroll for simulation
        """
        self.initial_bankroll = initial_bankroll
        self.results = {}
        
    def edge(self, p: float, odds: float) -> float:
        """
        Calculate edge using exact formula: Edge = p̂ ⋅ (O - 1) - (1 - p̂)
        
        Args:
            p: Model predicted probability (p̂)
            odds: Decimal odds (O)
            
        Returns:
            float: Edge value
        """
        b = odds - 1
        return p * b - (1 - p)
    
    def roi(self, total_profit: float, total_stakes: float) -> float:
        """
        Calculate ROI using exact formula: ROI = Total Profit / Total Stakes
        
        Args:
            total_profit: Total profit from bets
            total_stakes: Total amount staked
            
        Returns:
            float: ROI value
        """
        return total_profit / total_stakes if total_stakes > 0 else 0.0
    
    def kelly_fraction(self, p: float, odds: float) -> float:
        """
        Calculate Kelly fraction using exact formula: f* = (bp - q) / b
        
        Where:
        - b = O - 1
        - p = p̂ (model prediction)
        - q = 1 - p
        
        Args:
            p: Model predicted probability (p̂)
            odds: Decimal odds (O)
            
        Returns:
            float: Kelly fraction (capped at 0.0 for negative values)
        """
        b = odds - 1
        q = 1 - p
        kelly = (b * p - q) / b
        return max(kelly, 0.0)  # No negative betting
    
    def simulate_betting_strategy(self, 
                                predictions: np.ndarray, 
                                probabilities: np.ndarray, 
                                true_outcomes: np.ndarray,
                                odds: np.ndarray,
                                strategy: str = 'flat',
                                bet_size: float = 0.02,
                                kelly_cap: float = 0.1) -> Dict:
        """
        Simulate betting strategy performance.
        
        Args:
            predictions: Model predictions (0 or 1)
            probabilities: Model probability predictions
            true_outcomes: True game outcomes (0 or 1)
            odds: Decimal odds for each bet
            strategy: 'flat', 'kelly', or 'edge_based'
            bet_size: Fixed bet size for flat betting (as fraction of bankroll)
            kelly_cap: Maximum Kelly fraction to cap risk
            
        Returns:
            Dict: Simulation results
        """
        n_bets = len(predictions)
        bankroll = self.initial_bankroll
        bets_placed = []
        profits = []
        cumulative_bankroll = [bankroll]
        
        for i in range(n_bets):
            pred = predictions[i]
            prob = probabilities[i]
            outcome = true_outcomes[i]
            odd = odds[i]
            
            # Calculate edge
            edge_val = self.edge(prob, odd)
            
            # Determine bet size based on strategy
            if strategy == 'flat':
                bet_fraction = bet_size
            elif strategy == 'kelly':
                kelly_val = self.kelly_fraction(prob, odd)
                bet_fraction = min(kelly_val, kelly_cap)  # Cap Kelly for risk management
            elif strategy == 'edge_based':
                # Only bet when edge > 0
                if edge_val > 0:
                    bet_fraction = min(edge_val * 2, 0.05)  # Scale edge to bet size
                else:
                    bet_fraction = 0
            else:
                raise ValueError("Strategy must be 'flat', 'kelly', or 'edge_based'")
            
            # Place bet if strategy allows
            if bet_fraction > 0:
                bet_amount = bankroll * bet_fraction
                bets_placed.append(bet_amount)
                
                # Calculate profit/loss
                if outcome == 1:  # Win
                    profit = bet_amount * (odd - 1)
                else:  # Loss
                    profit = -bet_amount
                
                profits.append(profit)
                bankroll += profit
            else:
                bets_placed.append(0)
                profits.append(0)
            
            cumulative_bankroll.append(bankroll)
        
        # Calculate final metrics
        total_bets = len([b for b in bets_placed if b > 0])
        total_stakes = sum(bets_placed)
        total_profit = sum(profits)
        final_roi = self.roi(total_profit, total_stakes)
        
        # Win rate
        winning_bets = sum(1 for i, p in enumerate(profits) if p > 0 and bets_placed[i] > 0)
        win_rate = winning_bets / total_bets if total_bets > 0 else 0
        
        # Average edge
        edges = [self.edge(prob, odd) for prob, odd in zip(probabilities, odds)]
        avg_edge = np.mean(edges)
        
        results = {
            'strategy': strategy,
            'total_bets': total_bets,
            'total_stakes': total_stakes,
            'total_profit': total_profit,
            'final_bankroll': bankroll,
            'roi': final_roi,
            'win_rate': win_rate,
            'avg_edge': avg_edge,
            'cumulative_bankroll': cumulative_bankroll,
            'bets_placed': bets_placed,
            'profits': profits,
            'edges': edges
        }
        
        self.results[strategy] = results
        return results
    
    def compare_strategies(self, 
                          predictions: np.ndarray, 
                          probabilities: np.ndarray, 
                          true_outcomes: np.ndarray,
                          odds: np.ndarray,
                          bet_size: float = 0.02) -> pd.DataFrame:
        """
        Compare multiple betting strategies.
        
        Args:
            predictions: Model predictions
            probabilities: Model probabilities
            true_outcomes: True outcomes
            odds: Decimal odds
            bet_size: Fixed bet size for flat betting
            
        Returns:
            DataFrame: Comparison of all strategies
        """
        strategies = ['flat', 'kelly', 'edge_based']
        comparison_data = []
        
        for strategy in strategies:
            results = self.simulate_betting_strategy(
                predictions, probabilities, true_outcomes, odds, strategy, bet_size
            )
            
            comparison_data.append({
                'Strategy': strategy,
                'Total Bets': results['total_bets'],
                'Total Stakes': f"${results['total_stakes']:,.2f}",
                'Total Profit': f"${results['total_profit']:,.2f}",
                'Final Bankroll': f"${results['final_bankroll']:,.2f}",
                'ROI': f"{results['roi']:.2%}",
                'Win Rate': f"{results['win_rate']:.2%}",
                'Avg Edge': f"{results['avg_edge']:.3f}"
            })
        
        return pd.DataFrame(comparison_data)
    
    def plot_bankroll_evolution(self, strategies: List[str] = None):
        """
        Plot bankroll evolution for different strategies.
        
        Args:
            strategies: List of strategies to plot (default: all)
        """
        if strategies is None:
            strategies = list(self.results.keys())
        
        plt.figure(figsize=(12, 8))
        
        for strategy in strategies:
            if strategy in self.results:
                bankroll = self.results[strategy]['cumulative_bankroll']
                plt.plot(bankroll, label=f'{strategy.title()} Strategy', linewidth=2)
        
        plt.axhline(y=self.initial_bankroll, color='black', linestyle='--', alpha=0.7, label='Initial Bankroll')
        plt.xlabel('Bet Number')
        plt.ylabel('Bankroll ($)')
        plt.title('Bankroll Evolution by Betting Strategy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        os.makedirs('outputs/phase3/plots', exist_ok=True)
        plt.savefig('outputs/phase3/plots/bankroll_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_edge_distribution(self, strategies: List[str] = None):
        """
        Plot edge distribution for different strategies.
        
        Args:
            strategies: List of strategies to plot (default: all)
        """
        if strategies is None:
            strategies = list(self.results.keys())
        
        fig, axes = plt.subplots(1, len(strategies), figsize=(15, 5))
        if len(strategies) == 1:
            axes = [axes]
        
        for i, strategy in enumerate(strategies):
            if strategy in self.results:
                edges = self.results[strategy]['edges']
                profits = self.results[strategy]['profits']
                
                # Create scatter plot of edge vs profit
                axes[i].scatter(edges, profits, alpha=0.6)
                axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.7)
                axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.7)
                axes[i].set_xlabel('Edge')
                axes[i].set_ylabel('Profit/Loss ($)')
                axes[i].set_title(f'{strategy.title()} Strategy: Edge vs Profit')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('outputs/phase3/plots', exist_ok=True)
        plt.savefig('outputs/phase3/plots/edge_vs_profit.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_roi_report(self) -> str:
        """
        Generate comprehensive ROI report.
        
        Returns:
            str: Formatted ROI report
        """
        if not self.results:
            return "No simulation results available. Run simulate_betting_strategy first."
        
        report = "=" * 80 + "\n"
        report += "CBB BETTING ML SYSTEM - ROI SIMULATION REPORT\n"
        report += "=" * 80 + "\n\n"
        
        # Strategy comparison table
        report += "STRATEGY COMPARISON:\n"
        report += "-" * 50 + "\n"
        
        for strategy, results in self.results.items():
            report += f"\n{strategy.upper()} STRATEGY:\n"
            report += f"  Total Bets: {results['total_bets']}\n"
            report += f"  Total Stakes: ${results['total_stakes']:,.2f}\n"
            report += f"  Total Profit: ${results['total_profit']:,.2f}\n"
            report += f"  Final Bankroll: ${results['final_bankroll']:,.2f}\n"
            report += f"  ROI: {results['roi']:.2%}\n"
            report += f"  Win Rate: {results['win_rate']:.2%}\n"
            report += f"  Average Edge: {results['avg_edge']:.3f}\n"
        
        # Key insights
        report += "\n" + "=" * 80 + "\n"
        report += "KEY INSIGHTS:\n"
        report += "=" * 80 + "\n"
        
        # Find best performing strategy
        best_strategy = max(self.results.keys(), key=lambda x: self.results[x]['roi'])
        best_roi = self.results[best_strategy]['roi']
        
        report += f"• Best Performing Strategy: {best_strategy.title()} (ROI: {best_roi:.2%})\n"
        
        # Risk analysis
        for strategy, results in self.results.items():
            max_drawdown = self._calculate_max_drawdown(results['cumulative_bankroll'])
            report += f"• {strategy.title()} Max Drawdown: {max_drawdown:.2%}\n"
        
        # Formula verification
        report += "\n" + "=" * 80 + "\n"
        report += "FORMULA VERIFICATION:\n"
        report += "=" * 80 + "\n"
        report += "• Edge = p̂ ⋅ (O - 1) - (1 - p̂) ✓\n"
        report += "• ROI = Total Profit / Total Stakes ✓\n"
        report += "• Kelly = (bp - q) / b, where b = O - 1, p = p̂, q = 1 - p ✓\n"
        
        return report
    
    def _calculate_max_drawdown(self, bankroll_history: List[float]) -> float:
        """
        Calculate maximum drawdown from peak.
        
        Args:
            bankroll_history: List of bankroll values over time
            
        Returns:
            float: Maximum drawdown as percentage
        """
        peak = bankroll_history[0]
        max_drawdown = 0
        
        for value in bankroll_history:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def save_results(self, filepath: str):
        """
        Save simulation results to file.
        
        Args:
            filepath: Path to save results
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert results to DataFrame for easy saving
        results_df = pd.DataFrame()
        
        for strategy, results in self.results.items():
            strategy_df = pd.DataFrame({
                'strategy': [strategy] * len(results['bets_placed']),
                'bet_number': range(len(results['bets_placed'])),
                'bet_amount': results['bets_placed'],
                'profit_loss': results['profits'],
                'edge': results['edges'],
                'cumulative_bankroll': results['cumulative_bankroll'][1:]  # Skip initial
            })
            results_df = pd.concat([results_df, strategy_df], ignore_index=True)
        
        results_df.to_csv(filepath, index=False)
        print(f"✅ ROI simulation results saved to: {filepath}")
        
        # Save summary report
        report_path = filepath.replace('.csv', '_report.txt')
        with open(report_path, 'w') as f:
            f.write(self.generate_roi_report())
        print(f"✅ ROI simulation report saved to: {report_path}")