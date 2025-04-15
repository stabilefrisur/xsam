from attrs import define, field
from .asset import Asset

@define
class Trade:
    asset: Asset
    quantity: float
    reversion_pnl: float | None = field(default=None)
    break_even_pnl: float | None = field(default=None)
    safe_pnl: float | None = field(default=None)
    worst_pnl: float | None = field(default=None)

    def __attrs_post_init__(self):
        """Post-initialization processing to calculate safe and worst PnL."""
        self.reversion_pnl = self.calc_reversion_pnl()
        self.break_even_pnl = self.calc_break_even_pnl()
        self.safe_pnl = self.calc_safe_pnl()
        self.worst_pnl = self.calc_worst_pnl()

    @staticmethod
    def calc_pnl(quantity: float, ret: float) -> float:
        """Calculate the profit and loss (PnL) based on quantity and return.

        Args:
            quantity (float): Dollar amount of the trade.
            ret (float): Return of the asset in basis points.

        Returns:
            float: Profit and loss in dollars.
        """
        return quantity * ret / 1e4
    
    def calc_reversion_pnl(self) -> float:
        return self.calc_pnl(self.quantity, self.asset.reversion_ret)
    
    def calc_break_even_pnl(self) -> float:
        return self.calc_pnl(self.quantity, self.asset.break_even_ret)
    
    def calc_safe_pnl(self) -> float:
        return self.calc_pnl(self.quantity, self.asset.safe_ret)
    
    def calc_worst_pnl(self) -> float:
        return self.calc_pnl(self.quantity, self.asset.worst_ret)

