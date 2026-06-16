"""
NSE Transaction Cost Model
Calculates all transaction costs, taxes, and brokerage fees for the Indian market.
"""

from typing import Dict


class NSETransactionCostModel:
    """
    Computes all-in transaction costs for trading on the NSE:
    - Securities Transaction Tax (STT)
    - Stamp Duty
    - Exchange Transaction Charges
    - GST (18% on brokerage + exchange transaction charges)
    - SEBI Turnover Fees
    - Brokerage (Zerodha style)
    """

    def __init__(self, flat_brokerage: float = 20.0, pct_brokerage: float = 0.0003) -> None:
        """
        Args:
            flat_brokerage: Flat fee cap (e.g., ₹20 per executed order).
            pct_brokerage: Percentage rate (e.g., 0.03% = 0.0003).
        """
        self.flat_brokerage = flat_brokerage
        self.pct_brokerage = pct_brokerage

    def calculate_cost(
        self,
        price: float,
        quantity: float,
        side: str,         # 'buy' or 'sell'
        product_type: str  # 'delivery', 'intraday', 'futures', 'options'
    ) -> Dict[str, float]:
        """
        Calculate total transaction cost for an trade.
        """
        turnover = price * quantity
        side_lower = side.lower()
        product_lower = product_type.lower()

        # 1. Brokerage
        if product_lower == "delivery":
            brokerage = 0.0  # Zerodha delivery is free
        else:
            brokerage = min(self.flat_brokerage, turnover * self.pct_brokerage)

        # 2. STT (Securities Transaction Tax)
        stt = 0.0
        if product_lower == "delivery":
            stt = turnover * 0.001  # 0.1% on buy and sell
        elif product_lower == "intraday":
            if side_lower == "sell":
                stt = turnover * 0.00025  # 0.025% on sell side only
        elif product_lower == "futures":
            if side_lower == "sell":
                stt = turnover * 0.0001  # 0.01% on sell side
        elif product_lower == "options":
            if side_lower == "sell":
                stt = turnover * 0.0005  # 0.05% on sell side

        # 3. Stamp Duty (Applies only on BUY side)
        stamp_duty = 0.0
        if side_lower == "buy":
            if product_lower == "delivery":
                stamp_duty = turnover * 0.00015  # 0.015%
            elif product_lower == "intraday":
                stamp_duty = turnover * 0.00003  # 0.003%
            elif product_lower == "futures":
                stamp_duty = turnover * 0.00002  # 0.002%
            elif product_lower == "options":
                stamp_duty = turnover * 0.00003  # 0.003%

        # 4. Exchange Transaction Charges (NSE)
        if product_lower == "options":
            exchange_charges = turnover * 0.0005  # ~0.05% on premium
        else:
            exchange_charges = turnover * 0.0000325  # ~0.00325% on equity/futures

        # 5. SEBI Turnover Fees
        sebi_fees = turnover * 0.0000001  # ₹10 per crore (0.00001%)

        # 6. GST (18% on Brokerage + Exchange charges)
        gst = (brokerage + exchange_charges + sebi_fees) * 0.18

        total_taxes = stt + stamp_duty + exchange_charges + sebi_fees + gst
        total_cost = brokerage + total_taxes

        return {
            "turnover": turnover,
            "brokerage": brokerage,
            "stt": stt,
            "stamp_duty": stamp_duty,
            "exchange_charges": exchange_charges,
            "sebi_fees": sebi_fees,
            "gst": gst,
            "total_taxes": total_taxes,
            "total_cost": total_cost,
        }
