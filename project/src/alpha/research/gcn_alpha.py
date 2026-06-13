"""Game-theoretic and graph neural alpha components.

This module promotes the roadmap's GameStock and BCF-GCN ideas into concrete,
testable building blocks that do not depend on torch-geometric.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np
import pandas as pd


class InvestorType(Enum):
    """Coarse investor types from GameStock-style market modeling."""

    INSTITUTIONAL = "institutional"
    HOT_MONEY = "hot_money"
    RETAIL = "retail"


@dataclass(frozen=True)
class InvestorFlow:
    """Aggregated signed flow for one symbol and investor type."""

    symbol: str
    investor_type: InvestorType
    signed_volume: float
    price_impact: float


@dataclass(frozen=True)
class GameSignal:
    """Investor-game signal for one symbol."""

    symbol: str
    signal: float
    confidence: float
    institutional_flow: float
    hot_money_flow: float
    retail_flow: float
    regime: str


def classify_investor(
    trade_volume: float,
    price_impact: float,
    adv: float | None = None,
    institutional_participation: float = 0.02,
    hot_money_participation: float = 0.005,
) -> InvestorType:
    """Classify a trade into institutional, hot-money, or retail flow.

    `adv` lets the thresholds scale with symbol liquidity. Without ADV, sane
    absolute thresholds are used for Indian equity lots.
    """
    volume = abs(float(trade_volume))
    impact = abs(float(price_impact))
    institutional_threshold = max(50_000.0, (adv or 0.0) * institutional_participation)
    hot_money_threshold = max(10_000.0, (adv or 0.0) * hot_money_participation)

    if volume >= institutional_threshold and impact <= 0.0015:
        return InvestorType.INSTITUTIONAL
    if volume >= hot_money_threshold and impact >= 0.004:
        return InvestorType.HOT_MONEY
    return InvestorType.RETAIL


class GameStockAlpha:
    """Aggregate heterogeneous investor behavior into cross-sectional signals."""

    def __init__(self, reversal_threshold: float = 0.6) -> None:
        self.reversal_threshold = reversal_threshold

    def aggregate_flows(self, trades: pd.DataFrame, adv_by_symbol: dict[str, float] | None = None) -> list[InvestorFlow]:
        """Classify and aggregate a trade blotter.

        Required columns: `symbol`, `volume`, `price_impact`.
        Optional columns: `signed_volume`, `side`.
        """
        required = {"symbol", "volume", "price_impact"}
        missing = required - set(trades.columns)
        if missing:
            raise ValueError(f"trades missing required columns: {sorted(missing)}")

        adv_by_symbol = adv_by_symbol or {}
        buckets: dict[tuple[str, InvestorType], list[tuple[float, float]]] = {}

        for row in trades.itertuples(index=False):
            symbol = str(getattr(row, "symbol"))
            volume = float(getattr(row, "volume"))
            impact = float(getattr(row, "price_impact"))
            investor_type = classify_investor(volume, impact, adv_by_symbol.get(symbol))
            signed_volume = self._signed_volume(row, volume)
            buckets.setdefault((symbol, investor_type), []).append((signed_volume, impact))

        flows = []
        for (symbol, investor_type), values in buckets.items():
            signed = float(sum(v[0] for v in values))
            avg_impact = float(np.average([abs(v[1]) for v in values], weights=[abs(v[0]) for v in values]))
            flows.append(InvestorFlow(symbol, investor_type, signed, avg_impact))
        return flows

    def compute_signals(self, flows: Iterable[InvestorFlow]) -> pd.DataFrame:
        """Compute GameStock-style signals from aggregated flows."""
        rows = []
        by_symbol: dict[str, dict[InvestorType, float]] = {}
        for flow in flows:
            by_symbol.setdefault(flow.symbol, {})[flow.investor_type] = flow.signed_volume

        for symbol, flow_map in by_symbol.items():
            inst = flow_map.get(InvestorType.INSTITUTIONAL, 0.0)
            hot = flow_map.get(InvestorType.HOT_MONEY, 0.0)
            retail = flow_map.get(InvestorType.RETAIL, 0.0)
            scale = max(abs(inst) + abs(hot) + abs(retail), 1.0)

            inst_score = inst / scale
            hot_score = hot / scale
            retail_score = retail / scale
            crowding = abs(hot_score - inst_score)

            if hot_score > 0 and inst_score < 0 and crowding > self.reversal_threshold:
                regime = "hot_money_momentum_reversal_risk"
                signal = 0.5 * hot_score + inst_score
            elif inst_score > 0 and retail_score < 0:
                regime = "institutional_accumulation"
                signal = inst_score - 0.25 * retail_score
            elif retail_score > 0 and inst_score < 0:
                regime = "retail_chase_distribution"
                signal = inst_score - 0.5 * retail_score
            else:
                regime = "mixed_flow"
                signal = 0.6 * inst_score + 0.3 * hot_score - 0.1 * retail_score

            rows.append(
                GameSignal(
                    symbol=symbol,
                    signal=float(np.clip(signal, -1.0, 1.0)),
                    confidence=float(np.clip(crowding + abs(signal), 0.0, 1.0)),
                    institutional_flow=inst,
                    hot_money_flow=hot,
                    retail_flow=retail,
                    regime=regime,
                )
            )

        return pd.DataFrame([row.__dict__ for row in rows]).set_index("symbol") if rows else pd.DataFrame()

    def _signed_volume(self, row: object, volume: float) -> float:
        if hasattr(row, "signed_volume"):
            return float(getattr(row, "signed_volume"))
        side = str(getattr(row, "side", "buy")).lower()
        return volume if side in {"buy", "b", "1", "long"} else -volume


def correlation_graph(returns: pd.DataFrame, threshold: float = 0.35) -> pd.DataFrame:
    """Build a stock graph adjacency matrix from trailing return correlations."""
    if returns.empty:
        return pd.DataFrame()
    corr = returns.corr().fillna(0.0)
    adjacency = (corr.abs() >= threshold).astype(float)
    np.fill_diagonal(adjacency.values, 1.0)
    return adjacency


try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


if torch is not None:

    class DenseGraphConvolution(nn.Module):
        """Simple A_hat X W graph convolution."""

        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)

        def forward(self, x: "torch.Tensor", adjacency: "torch.Tensor") -> "torch.Tensor":
            adj = normalize_adjacency(adjacency)
            return adj @ self.linear(x)


    class BiLevelChaoticFusionGCN(nn.Module):
        """BCF-GCN approximation with chaotic center/width branches."""

        def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.0) -> None:
            super().__init__()
            self.gcn1 = DenseGraphConvolution(n_features, hidden)
            self.gcn2 = DenseGraphConvolution(hidden, hidden)
            self.dropout = nn.Dropout(dropout)
            self.center_head = nn.Linear(hidden, 1)
            self.width_head = nn.Linear(hidden, 1)
            self.gate = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
            self.raw_r = nn.Parameter(torch.tensor(3.8))
            self.raw_mu = nn.Parameter(torch.tensor(1.8))

        def forward(self, x: "torch.Tensor", adjacency: "torch.Tensor") -> dict[str, "torch.Tensor"]:
            h = torch.relu(self.gcn1(x, adjacency))
            h = self.dropout(torch.relu(self.gcn2(h, adjacency)))
            z = torch.sigmoid(self.center_head(h))
            r = self.raw_r.clamp(3.5, 4.0)
            mu = self.raw_mu.clamp(1.0, 2.0)

            chaotic_center = r * z * (1.0 - z)
            tent_width = mu * torch.minimum(z, 1.0 - z)
            gate = self.gate(h)

            center = gate * chaotic_center + (1.0 - gate) * z
            width = torch.nn.functional.softplus(self.width_head(h)) * (0.25 + tent_width)
            lower = center - width
            upper = center + width
            return {"prediction": center.squeeze(-1), "lower": lower.squeeze(-1), "upper": upper.squeeze(-1), "width": width.squeeze(-1)}


    def normalize_adjacency(adjacency: "torch.Tensor") -> "torch.Tensor":
        """Symmetrically normalize an adjacency matrix with self loops."""
        adj = adjacency.float()
        eye = torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
        adj = torch.maximum(adj, adj.T) + eye
        degree = adj.sum(dim=1).clamp_min(1e-12)
        inv_sqrt = torch.pow(degree, -0.5)
        return inv_sqrt[:, None] * adj * inv_sqrt[None, :]

else:

    class BiLevelChaoticFusionGCN:  # type: ignore[no-redef]
        """Placeholder that fails clearly when torch is not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError("BiLevelChaoticFusionGCN requires torch")
