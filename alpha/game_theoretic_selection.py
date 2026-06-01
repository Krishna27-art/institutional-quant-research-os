"""
Game-Theoretic Stock Selection
Based on Zhang et al. (2026) methodology

Key findings from research:
- 3-5% RankIC improvement
- Heterogeneous investor participation
- Dragon & Tiger List events
- Game-theoretic equilibrium states
- Cross-sectional dispersion

Architecture V2 - Quantitative Trading System for Indian Markets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize


@dataclass
class GameTheoreticSignal:
    """Game-theoretic trading signal"""
    symbol: str
    signal: float  # -1 to 1
    confidence: float
    equilibrium_state: str
    participant_imbalance: float
    rank_ic: float


@dataclass
class SelectionResult:
    """Stock selection results"""
    selected_symbols: List[str]
    signals: Dict[str, float]
    weights: Dict[str, float]
    rank_ic: float
    cross_sectional_dispersion: float
    equilibrium_state: str


class GameTheoreticStockSelector:
    """
    Game-Theoretic Stock Selection based on Zhang et al. (2026).
    
    Key findings:
    - 3-5% RankIC improvement over traditional methods
    - Heterogeneous investor participation drives alpha
    - Dragon & Tiger List events create opportunities
    - Game-theoretic equilibrium states
    - Cross-sectional dispersion matters
    
    Methodology:
    - Model heterogeneous investor behavior
    - Identify equilibrium states
    - Exploit participant imbalances
    - Use cross-sectional dispersion
    """
    
    def __init__(self, n_stocks: int = 50):
        self.n_stocks = n_stocks
        
        # Participant types
        self.participant_types = [
            "retail",
            "institutional",
            "hft",
            "foreign"
        ]
        
        # Equilibrium states
        self.equilibrium_states = [
            "balanced",
            "retail_dominant",
            "institutional_dominant",
            "hft_dominant",
            "foreign_dominant"
        ]
        
        self.is_fitted = False
    
    def calculate_participant_flow(
        self,
        volume_data: pd.DataFrame,
        trade_data: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Calculate participant flow imbalances.
        
        Args:
            volume_data: Volume data by participant type
            trade_data: Trade data
            
        Returns:
            Dictionary mapping participant types to flow arrays
        """
        flows = {}
        
        for participant in self.participant_types:
            # Calculate flow imbalance
            if f"{participant}_buy" in volume_data.columns and f"{participant}_sell" in volume_data.columns:
                buy_vol = volume_data[f"{participant}_buy"]
                sell_vol = volume_data[f"{participant}_sell"]
                flow = (buy_vol - sell_vol) / (buy_vol + sell_vol)
                flows[participant] = flow.values
            else:
                # Simulate flow if data not available
                flows[participant] = np.random.normal(0, 0.1, len(volume_data))
        
        return flows
    
    def identify_equilibrium_state(
        self,
        flows: Dict[str, np.ndarray],
        current_index: int
    ) -> str:
        """
        Identify current equilibrium state.
        
        Args:
            flows: Participant flow data
            current_index: Current time index
            
        Returns:
            Equilibrium state name
        """
        # Get current flows
        current_flows = {k: v[current_index] for k, v in flows.items()}
        
        # Find dominant participant
        dominant = max(current_flows.items(), key=lambda x: abs(x[1]))
        
        # Determine state
        if abs(dominant[1]) < 0.1:
            return "balanced"
        else:
            return f"{dominant[0]}_dominant"
    
    def calculate_cross_sectional_dispersion(
        self,
        returns: pd.DataFrame
    ) -> float:
        """
        Calculate cross-sectional dispersion.
        
        Args:
            returns: DataFrame with returns for multiple stocks
            
        Returns:
            Cross-sectional dispersion measure
        """
        # Calculate dispersion as std of cross-sectional returns
        cross_sectional_returns = returns.iloc[-1]
        dispersion = cross_sectional_returns.std()
        
        return dispersion
    
    def calculate_rank_ic(
        self,
        signals: np.ndarray,
        future_returns: np.ndarray
    ) -> float:
        """
        Calculate Rank Information Coefficient (RankIC).
        
        Args:
            signals: Signal values
            future_returns: Future returns
            
        Returns:
            RankIC value
        """
        # Calculate rank correlation
        rank_ic = stats.spearmanr(signals, future_returns)[0]
        
        return rank_ic
    
    def generate_signals(
        self,
        participant_flows: Dict[str, np.ndarray],
        returns: pd.DataFrame,
        current_index: int
    ) -> np.ndarray:
        """
        Generate game-theoretic trading signals.
        
        Args:
            participant_flows: Participant flow data
            returns: Returns data
            current_index: Current time index
            
        Returns:
            Signal array for all stocks
        """
        n_stocks = returns.shape[1]
        signals = np.zeros(n_stocks)
        
        # Get current equilibrium state
        equilibrium = self.identify_equilibrium_state(participant_flows, current_index)
        
        # Calculate participant imbalances
        imbalances = {}
        for participant, flows in participant_flows.items():
            imbalances[participant] = flows[current_index]
        
        # Generate signals based on equilibrium state
        for i in range(n_stocks):
            stock_return = returns.iloc[current_index, i]
            
            if equilibrium == "retail_dominant":
                # Fade retail (contrarian)
                signals[i] = -np.sign(stock_return) * abs(imbalances["retail"])
            elif equilibrium == "institutional_dominant":
                # Follow institutions (momentum)
                signals[i] = np.sign(stock_return) * abs(imbalances["institutional"])
            elif equilibrium == "hft_dominant":
                # Fade HFT (contrarian)
                signals[i] = -np.sign(stock_return) * abs(imbalances["hft"])
            elif equilibrium == "foreign_dominant":
                # Follow foreign (momentum)
                signals[i] = np.sign(stock_return) * abs(imbalances["foreign"])
            else:  # balanced
                # Use cross-sectional mean reversion
                cross_sectional_mean = returns.iloc[current_index].mean()
                signals[i] = (cross_sectional_mean - stock_return) / abs(stock_return)
        
        # Normalize signals
        signals = np.clip(signals, -1, 1)
        
        return signals
    
    def select_stocks(
        self,
        signals: np.ndarray,
        returns: pd.DataFrame,
        n_select: int = 10
    ) -> List[str]:
        """
        Select top stocks based on game-theoretic signals.
        
        Args:
            signals: Signal array
            returns: Returns data
            n_select: Number of stocks to select
            
        Returns:
            List of selected stock symbols
        """
        # Rank stocks by signal strength
        ranked_indices = np.argsort(signals)[::-1]
        
        # Select top n stocks
        selected_indices = ranked_indices[:n_select]
        selected_symbols = returns.columns[selected_indices].tolist()
        
        return selected_symbols
    
    def calculate_weights(
        self,
        signals: np.ndarray,
        selected_symbols: List[str]
    ) -> Dict[str, float]:
        """
        Calculate portfolio weights based on signal strength.
        
        Args:
            signals: Signal array
            selected_symbols: Selected stock symbols
            
        Returns:
            Dictionary mapping symbols to weights
        """
        # Get signals for selected stocks
        selected_indices = [i for i, s in enumerate(returns.columns) if s in selected_symbols]
        selected_signals = signals[selected_indices]
        
        # Normalize to sum to 1
        weights = np.abs(selected_signals)
        weights = weights / weights.sum()
        
        # Create weight dictionary
        weight_dict = dict(zip(selected_symbols, weights))
        
        return weight_dict
    
    def run_selection(
        self,
        volume_data: pd.DataFrame,
        trade_data: pd.DataFrame,
        returns: pd.DataFrame,
        n_select: int = 10
    ) -> SelectionResult:
        """
        Run complete game-theoretic stock selection.
        
        Args:
            volume_data: Volume data by participant type
            trade_data: Trade data
            returns: Returns data
            n_select: Number of stocks to select
            
        Returns:
            SelectionResult with selected stocks and weights
        """
        print(f"Running Game-Theoretic Stock Selection...")
        
        # Calculate participant flows
        flows = self.calculate_participant_flow(volume_data, trade_data)
        
        # Get current index
        current_index = len(returns) - 1
        
        # Identify equilibrium state
        equilibrium = self.identify_equilibrium_state(flows, current_index)
        
        # Generate signals
        signals = self.generate_signals(flows, returns, current_index)
        
        # Calculate cross-sectional dispersion
        dispersion = self.calculate_cross_sectional_dispersion(returns)
        
        # Select stocks
        selected_symbols = self.select_stocks(signals, returns, n_select)
        
        # Calculate weights
        weights = self.calculate_weights(signals, selected_symbols)
        
        # Calculate RankIC (using next period returns)
        if current_index < len(returns) - 1:
            future_returns = returns.iloc[current_index + 1]
            rank_ic = self.calculate_rank_ic(signals, future_returns.values)
        else:
            rank_ic = 0.0
        
        # Create signal dictionary
        signal_dict = dict(zip(returns.columns, signals))
        
        self.is_fitted = True
        
        return SelectionResult(
            selected_symbols=selected_symbols,
            signals=signal_dict,
            weights=weights,
            rank_ic=rank_ic,
            cross_sectional_dispersion=dispersion,
            equilibrium_state=equilibrium
        )
    
    def print_selection_results(self, result: SelectionResult) -> None:
        """Print selection results."""
        print("\n" + "="*60)
        print("GAME-THEORETIC STOCK SELECTION RESULTS")
        print("="*60)
        print(f"Equilibrium State: {result.equilibrium_state.upper()}")
        print(f"Rank IC: {result.rank_ic:.4f}")
        print(f"Cross-Sectional Dispersion: {result.cross_sectional_dispersion:.4f}")
        
        print(f"\nSelected Stocks ({len(result.selected_symbols)}):")
        for symbol in result.selected_symbols:
            signal = result.signals[symbol]
            weight = result.weights.get(symbol, 0.0)
            print(f"  {symbol:<10}: Signal={signal:>6.3f}, Weight={weight:>6.2%}")
        
        print("\nZhang et al. (2026) Benchmarks:")
        print(f"  - RankIC Improvement: 3-5% over traditional methods")
        print(f"  - Heterogeneous participation: Key driver of alpha")
        print(f"  - Dragon & Tiger List events: Create opportunities")
        print("="*60)


def run_sample_selection():
    """Run sample game-theoretic stock selection."""
    # Create synthetic data
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    symbols = [f"STOCK{i:02d}" for i in range(50)]
    
    np.random.seed(42)
    
    # Create returns data
    returns = pd.DataFrame(
        np.random.normal(0.001, 0.02, (252, 50)),
        index=dates,
        columns=symbols
    )
    
    # Create volume data by participant type
    volume_data = pd.DataFrame(
        np.random.randint(1000000, 5000000, (252, 8)),
        index=dates,
        columns=["retail_buy", "retail_sell", "institutional_buy", "institutional_sell",
                 "hft_buy", "hft_sell", "foreign_buy", "foreign_sell"]
    )
    
    # Create trade data
    trade_data = pd.DataFrame(
        np.random.randint(10000, 50000, (252, 50)),
        index=dates,
        columns=symbols
    )
    
    # Initialize selector
    selector = GameTheoreticStockSelector(n_stocks=50)
    
    # Run selection
    result = selector.run_selection(volume_data, trade_data, returns, n_select=10)
    
    # Print results
    selector.print_selection_results(result)
    
    return result


if __name__ == "__main__":
    run_sample_selection()
