"""
Crisis Simulator
Stress testing against historical market crises (COVID, Adani, Russia-Ukraine, etc.)
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Callable
import numpy as np
import pandas as pd


class CrisisScenario(Enum):
    """Predefined crisis scenarios for stress testing"""
    COVID_2020 = "COVID_2020"
    ADANI_2023 = "ADANI_2023"
    RUSSIA_UKRAINE_2022 = "RUSSIA_UKRAINE_2022"
    FLASH_CRASH_2015 = "FLASH_CRASH_2015"
    RATE_HIKE_2022 = "RATE_HIKE_2022"
    LIQUIDITY_CRISIS = "LIQUIDITY_CRISIS"


@dataclass
class PassCriteria:
    """Criteria for passing crisis simulation"""
    max_drawdown_threshold: float = 0.25  # 25% max drawdown
    max_daily_loss_threshold: float = 0.05  # 5% max daily loss
    max_var_violation_rate: float = 0.05  # 5% VaR violation rate
    
    def to_dict(self) -> Dict:
        return {
            "max_drawdown_threshold": self.max_drawdown_threshold,
            "max_daily_loss_threshold": self.max_daily_loss_threshold,
            "max_var_violation_rate": self.max_var_violation_rate,
        }


@dataclass
class SimulationResult:
    """Results of crisis simulation for a strategy"""
    strategy_id: str
    scenario: CrisisScenario
    start_date: date
    end_date: date
    simulated: bool = False
    
    # Performance metrics
    total_return: float = 0.0
    max_drawdown: float = 0.0
    max_daily_loss: float = 0.0
    var_violation_rate: float = 0.0
    daily_returns: List[float] = field(default_factory=list)
    
    # Pass/fail
    passed: bool = False
    failures: List[str] = field(default_factory=list)
    
    # Metadata
    simulated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "strategy_id": self.strategy_id,
            "scenario": self.scenario.value,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "simulated": self.simulated,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "max_daily_loss": self.max_daily_loss,
            "var_violation_rate": self.var_violation_rate,
            "passed": self.passed,
            "failures": self.failures,
            "simulated_at": self.simulated_at.isoformat(),
        }


class CrisisSimulator:
    """
    Simulates strategy performance during historical crisis scenarios.
    Tests against pass criteria to ensure strategy resilience.
    """
    
    # Predefined crisis windows (start_date, end_date, description)
    CRISIS_WINDOWS = {
        CrisisScenario.COVID_2020: (
            date(2020, 3, 1),
            date(2020, 4, 30),
            "NIFTY dropped 40% in March 2020"
        ),
        CrisisScenario.ADANI_2023: (
            date(2023, 1, 24),
            date(2023, 2, 15),
            "Adani stocks crashed 70% after short seller report"
        ),
        CrisisScenario.RUSSIA_UKRAINE_2022: (
            date(2022, 2, 24),
            date(2022, 3, 15),
            "NIFTY gap down 5%, volatility spike"
        ),
        CrisisScenario.FLASH_CRASH_2015: (
            date(2015, 10, 5),
            date(2015, 10, 5),
            "5% drop in 10 minutes (synthetic)"
        ),
        CrisisScenario.RATE_HIKE_2022: (
            date(2022, 3, 1),
            date(2022, 6, 30),
            "200bps increase in 3 months"
        ),
        CrisisScenario.LIQUIDITY_CRISIS: (
            date(2020, 3, 23),
            date(2020, 3, 23),
            "Zero volume for 1 hour"
        ),
    }
    
    def __init__(self, pass_criteria: Optional[PassCriteria] = None):
        self.pass_criteria = pass_criteria or PassCriteria()
        self.simulation_results: Dict[str, List[SimulationResult]] = {}
        self.data_loader: Optional[Callable] = None
    
    def set_data_loader(self, data_loader: Callable) -> None:
        """
        Set data loader function for fetching historical market data.
        
        Args:
            data_loader: Function that takes (start_date, end_date) and returns DataFrame
        """
        self.data_loader = data_loader
    
    def get_crisis_window(self, scenario: CrisisScenario) -> tuple[date, date, str]:
        """Get crisis window for a scenario"""
        return self.CRISIS_WINDOWS[scenario]
    
    def simulate_strategy(
        self,
        strategy_id: str,
        strategy_function: Callable,
        scenario: CrisisScenario,
        parameters: Optional[Dict] = None
    ) -> SimulationResult:
        """
        Simulate a strategy during a crisis scenario.
        
        Args:
            strategy_id: Strategy identifier
            strategy_function: Function that takes market data and parameters, returns returns
            scenario: Crisis scenario to simulate
            parameters: Strategy parameters
        
        Returns:
            SimulationResult with performance metrics
        """
        start_date, end_date, description = self.get_crisis_window(scenario)
        
        result = SimulationResult(
            strategy_id=strategy_id,
            scenario=scenario,
            start_date=start_date,
            end_date=end_date
        )
        
        try:
            # Load market data for crisis window
            if self.data_loader is None:
                raise ValueError("Data loader not set. Call set_data_loader() first.")
            
            market_data = self.data_loader(start_date, end_date)
            
            # Run strategy simulation
            daily_returns = strategy_function(market_data, parameters or {})
            result.daily_returns = daily_returns
            result.simulated = True
            
            # Calculate metrics
            returns_array = np.array(daily_returns)
            result.total_return = np.sum(returns_array)
            
            # Max drawdown
            cumulative_returns = np.cumprod(1 + returns_array)
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / peak
            result.max_drawdown = abs(np.min(drawdown))
            
            # Max daily loss
            result.max_daily_loss = abs(np.min(returns_array))
            
            # VaR violation rate (simplified: count days with loss < -2%)
            var_threshold = -0.02
            violations = np.sum(returns_array < var_threshold)
            result.var_violation_rate = violations / len(returns_array) if len(returns_array) > 0 else 0
            
            # Check pass criteria
            result.passed, result.failures = self._check_pass_criteria(result)
            
        except Exception as e:
            result.failures.append(f"Simulation failed: {str(e)}")
            result.passed = False
        
        # Store result
        if strategy_id not in self.simulation_results:
            self.simulation_results[strategy_id] = []
        self.simulation_results[strategy_id].append(result)
        
        return result
    
    def _check_pass_criteria(self, result: SimulationResult) -> tuple[bool, List[str]]:
        """Check if simulation result meets pass criteria"""
        failures = []
        
        if result.max_drawdown > self.pass_criteria.max_drawdown_threshold:
            failures.append(
                f"Max drawdown {result.max_drawdown:.2%} exceeds threshold "
                f"{self.pass_criteria.max_drawdown_threshold:.2%}"
            )
        
        if result.max_daily_loss > self.pass_criteria.max_daily_loss_threshold:
            failures.append(
                f"Max daily loss {result.max_daily_loss:.2%} exceeds threshold "
                f"{self.pass_criteria.max_daily_loss_threshold:.2%}"
            )
        
        if result.var_violation_rate > self.pass_criteria.max_var_violation_rate:
            failures.append(
                f"VaR violation rate {result.var_violation_rate:.2%} exceeds threshold "
                f"{self.pass_criteria.max_var_violation_rate:.2%}"
            )
        
        return len(failures) == 0, failures
    
    def run_all_scenarios(
        self,
        strategy_id: str,
        strategy_function: Callable,
        parameters: Optional[Dict] = None
    ) -> Dict[str, SimulationResult]:
        """
        Run simulation for all predefined crisis scenarios.
        
        Args:
            strategy_id: Strategy identifier
            strategy_function: Strategy function
            parameters: Strategy parameters
        
        Returns:
            Dictionary mapping scenario name to SimulationResult
        """
        results = {}
        
        for scenario in CrisisScenario:
            result = self.simulate_strategy(
                strategy_id=strategy_id,
                strategy_function=strategy_function,
                scenario=scenario,
                parameters=parameters
            )
            results[scenario.value] = result
        
        return results
    
    def get_simulation_results(
        self,
        strategy_id: str
    ) -> List[Dict]:
        """Get all simulation results for a strategy"""
        if strategy_id not in self.simulation_results:
            return []
        
        return [r.to_dict() for r in self.simulation_results[strategy_id]]
    
    def get_failed_scenarios(
        self,
        strategy_id: str
    ) -> List[Dict]:
        """Get scenarios where strategy failed"""
        if strategy_id not in self.simulation_results:
            return []
        
        return [
            r.to_dict()
            for r in self.simulation_results[strategy_id]
            if not r.passed
        ]
    
    def generate_report(
        self,
        strategy_id: str
    ) -> Dict:
        """
        Generate comprehensive crisis simulation report.
        
        Args:
            strategy_id: Strategy identifier
        
        Returns:
            Report with summary and recommendations
        """
        results = self.simulation_results.get(strategy_id, [])
        
        if not results:
            return {
                "strategy_id": strategy_id,
                "status": "No simulations run",
                "recommendations": ["Run crisis simulations first"]
            }
        
        total_scenarios = len(results)
        passed_scenarios = sum(1 for r in results if r.passed)
        failed_scenarios = total_scenarios - passed_scenarios
        
        # Identify worst-performing scenarios
        worst_scenarios = sorted(
            results,
            key=lambda r: r.max_drawdown,
            reverse=True
        )[:3]
        
        # Generate recommendations
        recommendations = []
        
        if failed_scenarios > 0:
            recommendations.append(
                f"Strategy failed {failed_scenarios} out of {total_scenarios} crisis scenarios. "
                "Consider adding crisis-override rules (e.g., stop trading when VIX > 35)."
            )
        
        if any(r.max_drawdown > 0.20 for r in results):
            recommendations.append(
                "Strategy experienced significant drawdowns (>20%) during crises. "
                "Consider reducing position sizes or adding stop-loss mechanisms."
            )
        
        if passed_scenarios == total_scenarios:
            recommendations.append(
                "Strategy passed all crisis scenarios. Ready for live deployment."
            )
        
        return {
            "strategy_id": strategy_id,
            "total_scenarios": total_scenarios,
            "passed_scenarios": passed_scenarios,
            "failed_scenarios": failed_scenarios,
            "pass_rate": passed_scenarios / total_scenarios if total_scenarios > 0 else 0,
            "worst_scenarios": [r.to_dict() for r in worst_scenarios],
            "recommendations": recommendations,
            "generated_at": datetime.now().isoformat(),
        }
    
    def clear_results(self, strategy_id: Optional[str] = None) -> None:
        """Clear simulation results, optionally filtered by strategy"""
        if strategy_id is None:
            self.simulation_results.clear()
        else:
            self.simulation_results.pop(strategy_id, None)


def synthetic_crisis_data_generator(
    scenario: CrisisScenario,
    start_date: date,
    end_date: date
) -> pd.DataFrame:
    """
    Generate synthetic market data for crisis scenarios (for testing).
    
    Args:
        scenario: Crisis scenario
        start_date: Start date
        end_date: End date
    
    Returns:
        DataFrame with synthetic OHLCV data
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    if scenario == CrisisScenario.COVID_2020:
        # 40% drop over 30 days
        returns = np.linspace(-0.02, -0.01, n_days) + np.random.normal(0, 0.03, n_days)
    elif scenario == CrisisScenario.ADANI_2023:
        # 70% drop over 20 days
        returns = np.linspace(-0.05, -0.02, n_days) + np.random.normal(0, 0.05, n_days)
    elif scenario == CrisisScenario.RUSSIA_UKRAINE_2022:
        # 5% gap down, then volatility
        returns = np.concatenate([
            [-0.05],
            np.random.normal(0, 0.02, n_days - 1)
        ])
    elif scenario == CrisisScenario.FLASH_CRASH_2015:
        # 5% drop in 10 minutes (single day)
        returns = np.array([-0.05] + [0.0] * (n_days - 1))
    elif scenario == CrisisScenario.RATE_HIKE_2022:
        # Gradual decline with volatility
        returns = np.linspace(-0.005, -0.002, n_days) + np.random.normal(0, 0.015, n_days)
    else:
        # Default: random walk
        returns = np.random.normal(0, 0.02, n_days)
    
    # Generate price series
    prices = 100 * np.cumprod(1 + returns)
    
    # Create OHLCV
    data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, n_days)),
        'high': prices * (1 + np.random.uniform(0, 0.01, n_days)),
        'low': prices * (1 + np.random.uniform(-0.01, 0, n_days)),
        'close': prices,
        'volume': np.random.uniform(1e6, 5e6, n_days),
    })
    
    return data
