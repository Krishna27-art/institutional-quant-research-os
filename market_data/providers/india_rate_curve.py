"""
India Interest Rate Curve
=========================
Provides market-observable interest rates for Indian markets to replace
hardcoded OIS rates in options pricing and carry calculations.

India has no liquid OIS market. Use these sources in priority order:

  1. PRIMARY — Implied funding from NIFTY futures basis (always available)
     r_implied = ln(F/S) * (365/DTE)

  2. SECONDARY — MIBOR overnight rate from RBI DBIE API
     https://dbie.rbi.org.in/

  3. TERTIARY — G-Sec benchmark yield interpolation from NSE daily quotes

The implied funding rate (method 1) is the most reliable because it is
directly observable from live market prices and arbitrage-free by construction.
"""

from __future__ import annotations

import logging
import json
import time
from typing import Dict, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

import numpy as np

logger = logging.getLogger(__name__)


class IndiaRateCurve:
    """
    India interest rate curve — replaces hardcoded OIS in options pricing.

    Usage
    -----
    curve = IndiaRateCurve()

    # Best: derive from futures (no API needed)
    r = curve.get_implied_funding_rate(spot=22000, futures_price=22150, days_to_expiry=30)

    # Fallback: MIBOR from RBI
    r = curve.get_mibor_overnight()

    # For discounting option payoffs:
    df = curve.get_discount_factor(tenor_days=30)
    """

    MIBOR_CACHE_TTL_SECONDS = 3600       # Refresh MIBOR at most once per hour
    GSEC_TENORS = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 14, 30]   # Years

    # RBI DBIE API — returns JSON with MIBOR overnight rates
    # The actual endpoint for the JSON feed:
    _RBI_MIBOR_URL = (
        "https://dbie.rbi.org.in/DBIE/dbie.rbi?site=publications"
        "&type=1600&assetId=71"
    )
    _REQUEST_TIMEOUT_SECONDS = 3

    def __init__(self):
        self._mibor_cache: Optional[float] = None
        self._mibor_cache_ts: float = 0.0
        self._gsec_cache: Dict[float, float] = {}
        # Conservative fallback = RBI repo rate (updated manually as needed)
        self._last_known_mibor: float = 0.065   # 6.5%
        self._last_known_repo: float = 0.065    # 6.5%

    # ------------------------------------------------------------------
    # Primary: implied funding from futures basis
    # ------------------------------------------------------------------

    def get_implied_funding_rate(
        self,
        spot: float,
        futures_price: float,
        days_to_expiry: int,
    ) -> float:
        """
        Derive funding rate from NIFTY futures basis (continuously compounded).

        Formula: r = ln(F / S) / (DTE / 365)

        This is arbitrage-free and directly observable from market prices.
        It implicitly captures dividends, repo rates, and any convenience
        yield embedded in the futures contract.

        Parameters
        ----------
        spot : float
            Current spot index level (e.g., NIFTY50 = 22000).
        futures_price : float
            Current futures price for the relevant expiry.
        days_to_expiry : int
            Calendar days to futures expiry.

        Returns
        -------
        float : Annualised continuously-compounded rate (e.g., 0.065 for 6.5%).
        """
        if days_to_expiry <= 0:
            logger.warning("days_to_expiry <= 0, returning last known MIBOR")
            return self._last_known_mibor
        if spot <= 0 or futures_price <= 0:
            logger.warning("Invalid spot/futures price, returning last known MIBOR")
            return self._last_known_mibor

        tau = days_to_expiry / 365.0
        r = np.log(futures_price / spot) / tau
        # Sanity check: implied rate should be between -5% and 20%
        if not (-0.05 <= r <= 0.20):
            logger.warning(
                "Implied funding rate %.2f%% looks unreasonable for "
                "spot=%.2f futures=%.2f DTE=%d; falling back to MIBOR.",
                r * 100, spot, futures_price, days_to_expiry
            )
            return self._last_known_mibor

        self._last_known_mibor = r   # Update running estimate
        return float(r)

    # ------------------------------------------------------------------
    # Secondary: MIBOR overnight from RBI DBIE
    # ------------------------------------------------------------------

    def get_mibor_overnight(self) -> float:
        """
        Fetch the MIBOR overnight rate from the RBI DBIE web service.

        Returns the rate as a decimal (e.g., 0.065 for 6.5%).
        Falls back to the last known rate on any network or parse error.
        Cache TTL = 1 hour to avoid hammering the RBI server.
        """
        now = time.time()
        if (
            self._mibor_cache is not None
            and now - self._mibor_cache_ts < self.MIBOR_CACHE_TTL_SECONDS
        ):
            return self._mibor_cache

        try:
            req = Request(
                self._RBI_MIBOR_URL,
                headers={"User-Agent": "Mozilla/5.0 (InstitutionalQuant/1.0)"},
            )
            with urlopen(req, timeout=self._REQUEST_TIMEOUT_SECONDS) as resp:
                content = resp.read().decode("utf-8", errors="replace")

            # The RBI DBIE page returns HTML; parse the most recent MIBOR value.
            # MIBOR overnight typically appears as a number near "OVERNIGHT"
            # in the table. Use a simple heuristic search.
            import re
            # Pattern: look for a decimal like 6.50 or 6.75 near "OVERNIGHT"
            overnight_section = content.upper()
            idx = overnight_section.find("OVERNIGHT")
            if idx == -1:
                raise ValueError("Could not find OVERNIGHT section in RBI DBIE response")

            snippet = content[idx: idx + 200]
            matches = re.findall(r'\b(\d{1,2}\.\d{2})\b', snippet)
            if not matches:
                raise ValueError("No rate value found near OVERNIGHT keyword")

            rate_pct = float(matches[0])
            if not (1.0 <= rate_pct <= 20.0):
                raise ValueError(f"Rate {rate_pct} out of plausible range")

            rate = rate_pct / 100.0
            self._mibor_cache = rate
            self._mibor_cache_ts = now
            self._last_known_mibor = rate
            logger.info("MIBOR overnight fetched from RBI: %.2f%%", rate_pct)
            return rate

        except (URLError, HTTPError, OSError) as exc:
            logger.warning(
                "Network error fetching MIBOR from RBI DBIE: %s. "
                "Using last known rate %.2f%%.", exc, self._last_known_mibor * 100
            )
            return self._last_known_mibor

        except Exception as exc:
            logger.warning(
                "Failed to parse MIBOR from RBI DBIE: %s. "
                "Using last known rate %.2f%%.", exc, self._last_known_mibor * 100
            )
            return self._last_known_mibor

    # ------------------------------------------------------------------
    # Tertiary: G-Sec yield interpolation
    # ------------------------------------------------------------------

    def get_gsec_yield(self, tenor_years: float) -> float:
        """
        Get G-Sec yield for a given tenor via linear interpolation of
        NSE benchmark yields.

        NSE publishes G-Sec prices daily (closing):
        https://www.nseindia.com/market-data/government-securities

        Since fetching live G-Sec prices requires login/scraping, this
        method uses a simple approximate term structure:
            y(t) = MIBOR + term_premium(t)
        where term_premium is based on the typical India yield curve shape.

        For production use, replace with a live G-Sec data feed.

        Parameters
        ----------
        tenor_years : float
            Tenor in years (e.g., 0.25 = 3 months, 10 = 10-year).

        Returns
        -------
        float : Yield as decimal (continuously compounded).
        """
        mibor = self.get_mibor_overnight()

        # Approximate term premium schedule (basis points above overnight)
        # Based on typical India yield curve shape (as of 2024–2026)
        term_premium_bps = {
            0.08:  0,     # overnight
            0.25: 10,     # 3 months
            0.50: 20,     # 6 months
            1.0:  30,     # 1 year
            2.0:  45,     # 2 years
            3.0:  55,     # 3 years
            5.0:  65,     # 5 years
            7.0:  70,     # 7 years
            10.0: 75,     # 10 years
            14.0: 80,     # 14 years
            30.0: 85,     # 30 years
        }

        tenors = sorted(term_premium_bps.keys())
        if tenor_years <= tenors[0]:
            premium = term_premium_bps[tenors[0]]
        elif tenor_years >= tenors[-1]:
            premium = term_premium_bps[tenors[-1]]
        else:
            # Linear interpolation
            for i in range(len(tenors) - 1):
                t_lo, t_hi = tenors[i], tenors[i + 1]
                if t_lo <= tenor_years <= t_hi:
                    w = (tenor_years - t_lo) / (t_hi - t_lo)
                    premium = (
                        term_premium_bps[t_lo] * (1 - w)
                        + term_premium_bps[t_hi] * w
                    )
                    break

        return float(mibor + premium / 10000.0)

    # ------------------------------------------------------------------
    # Discount factor
    # ------------------------------------------------------------------

    def get_discount_factor(
        self,
        tenor_days: int,
        method: str = "futures",
        spot: float = 0.0,
        futures_price: float = 0.0,
    ) -> float:
        """
        Return discount factor exp(-r * T) for given tenor.

        Parameters
        ----------
        tenor_days : int
            Number of calendar days.
        method : str
            'futures' — use implied funding rate (requires spot + futures_price)
            'mibor'   — use MIBOR overnight from RBI
            'gsec'    — use G-Sec curve interpolation
        spot, futures_price : float
            Required only when method='futures'.

        Returns
        -------
        float : Discount factor in (0, 1].
        """
        tau = tenor_days / 365.0
        if method == "futures" and spot > 0 and futures_price > 0:
            r = self.get_implied_funding_rate(spot, futures_price, tenor_days)
        elif method == "gsec":
            r = self.get_gsec_yield(tau)
        else:
            r = self.get_mibor_overnight()

        return float(np.exp(-r * tau))

    # ------------------------------------------------------------------
    # Zero-curve bootstrap
    # ------------------------------------------------------------------

    def bootstrap_zero_curve(
        self, gsec_quotes: Dict[float, float]
    ) -> Dict[float, float]:
        """
        Bootstrap a zero-coupon yield curve from G-Sec par yields.

        Standard iterative bootstrap assuming annual coupon payments.
        For tenors < 1 year, the par yield IS the zero rate (no coupons).

        Parameters
        ----------
        gsec_quotes : Dict[float, float]
            {tenor_years: par_yield_decimal}
            e.g., {0.5: 0.068, 1.0: 0.069, 2.0: 0.070, 5.0: 0.072}

        Returns
        -------
        Dict[float, float] : {tenor_years: zero_rate_decimal}
        """
        tenors = sorted(gsec_quotes.keys())
        zero_rates: Dict[float, float] = {}

        for t in tenors:
            par_yield = gsec_quotes[t]

            if t <= 1.0:
                # Short tenor: par yield ≈ zero rate (no intermediate coupons)
                zero_rates[t] = par_yield
                continue

            # Bootstrap: price of par bond = 1 (by definition of par yield)
            # sum(coupon * df(t_i)) + 1 * df(T) = 1
            coupon = par_yield
            n_periods = int(t)   # Annual coupons
            coupon_times = [float(i) for i in range(1, n_periods)]
            coupon_times_with_face = coupon_times + [t]

            # Sum of discounted coupons using already-bootstrapped zero rates
            pv_coupons = 0.0
            for ct in coupon_times:
                if ct in zero_rates:
                    z = zero_rates[ct]
                elif zero_rates:
                    # Interpolate
                    known_t = sorted(zero_rates.keys())
                    z = float(np.interp(ct, known_t, [zero_rates[k] for k in known_t]))
                else:
                    z = par_yield
                pv_coupons += coupon * np.exp(-z * ct)

            # Solve for zero rate at maturity t:
            # coupon * sum(df(t_i)) + 1 * df(T) = 1
            # df(T) = (1 - pv_coupons) / (1 + coupon)  [approx for last period]
            # More precisely: df(T) = (1 - pv_coupons) / (1 + coupon)
            remaining = 1.0 - pv_coupons
            face_plus_coupon = 1.0 + coupon
            if remaining <= 0 or face_plus_coupon <= 0:
                zero_rates[t] = par_yield   # fallback
                continue
            df_T = remaining / face_plus_coupon
            if df_T <= 0:
                zero_rates[t] = par_yield
                continue
            zero_rates[t] = float(-np.log(df_T) / t)

        logger.info(
            "Bootstrapped zero curve: %d tenors from %.2f%% to %.2f%%",
            len(zero_rates),
            min(zero_rates.values()) * 100,
            max(zero_rates.values()) * 100,
        )
        return zero_rates
