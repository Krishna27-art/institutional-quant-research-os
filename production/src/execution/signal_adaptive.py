"""Signal-adaptive quoting inspired by Yu's optimal execution setup."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuoteDecision:
    """Execution quote depths from mid price."""

    bid_depth: float
    ask_depth: float
    urgency: float
    participation_rate: float


@dataclass(slots=True)
class SignalAdaptiveExecutor:
    """Compute quote depths from signal, inventory, volatility, and time left."""

    kappa: float = 1.5
    a: float = 0.0
    b: float = 1.0
    gamma: float = 0.05
    max_participation: float = 0.2

    def compute_quote(
        self,
        remaining_inventory: float,
        alpha_signal: float,
        time_left_fraction: float,
        volatility: float,
        inventory_limit: float | None = None,
    ) -> QuoteDecision:
        """Return bid/ask depth in price units or bps-equivalent units.

        Positive inventory means we need to sell; positive alpha means upward
        pressure, so selling can be less urgent and buying gets more aggressive.
        """
        inventory = float(remaining_inventory)
        signal = float(np.clip(alpha_signal, -1.0, 1.0))
        time_left = float(np.clip(time_left_fraction, 1e-6, 1.0))
        vol = max(float(volatility), 1e-8)
        limit = abs(float(inventory_limit)) if inventory_limit else max(abs(inventory), 1.0)

        inventory_pressure = np.clip(inventory / limit, -1.0, 1.0)
        urgency = np.clip(abs(inventory_pressure) / time_left, 0.0, 1.0)
        risk_term = self.gamma * vol * (1.0 + urgency)
        base_depth = (1.0 / max(self.b * self.gamma, 1e-8)) * np.log((self.kappa + self.b * self.gamma) / self.kappa)
        base_depth += self.a / max(self.b, 1e-8)
        base_depth += risk_term

        buy_aggression = np.clip(signal - inventory_pressure, -1.0, 1.0)
        sell_aggression = np.clip(-signal + inventory_pressure, -1.0, 1.0)

        bid_depth = base_depth * (1.0 - 0.5 * buy_aggression + 0.5 * urgency)
        ask_depth = base_depth * (1.0 - 0.5 * sell_aggression + 0.5 * urgency)
        participation = self.max_participation * (0.25 + 0.75 * urgency) * (0.5 + 0.5 * abs(signal))

        return QuoteDecision(
            bid_depth=float(max(bid_depth, 0.0)),
            ask_depth=float(max(ask_depth, 0.0)),
            urgency=float(urgency),
            participation_rate=float(np.clip(participation, 0.0, self.max_participation)),
        )
