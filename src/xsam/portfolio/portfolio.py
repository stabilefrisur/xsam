from attrs import define, field
from .trade import Trade


@define
class Portfolio:
    trades: list[Trade]
    weights: list[float] | None = field(default=None)
    # Modelled return and risk
    carry: float | None = field(default=None)
    value: float | None = field(default=None)
    spread_shock: float | None = field(default=None)
    shock_loss: float | None = field(default=None)
    # Market data
    price: float | None = field(default=None)
    spread_duration: float | None = field(default=None)
    spread: float | None = field(default=None)
    rolldown: float | None = field(default=None)
    expected_loss: float | None = field(default=None)
    # Calculated spreads
    reversion_spread: float | None = field(default=None)
    break_even_spread: float | None = field(default=None)
    safe_spread: float | None = field(default=None)
    shocked_spread: float | None = field(default=None)
    worst_spread: float | None = field(default=None)
    # Calculated PnL
    reversion_pnl: float | None = field(default=None)
    break_even_pnl: float | None = field(default=None)
    safe_pnl: float | None = field(default=None)
    worst_pnl: float | None = field(default=None)
    # Risk capacity
    capacity: float | None = field(default=None)

    def __attrs_post_init__(self):
        self.weights = self.calc_weights()
        # Calculate portfolio metrics
        self.carry = self.calc_portfolio_metric(self.trades, self.weights, "carry")
        self.value = self.calc_portfolio_metric(self.trades, self.weights, "value")
        self.spread_shock = self.calc_portfolio_metric(self.trades, self.weights, "spread_shock")
        self.shock_loss = self.calc_portfolio_metric(self.trades, self.weights, "shock_loss")
        self.price = self.calc_portfolio_metric(self.trades, self.weights, "price")
        self.spread_duration = self.calc_portfolio_metric(self.trades, self.weights, "spread_duration")
        self.spread = self.calc_portfolio_metric(self.trades, self.weights, "spread")
        self.rolldown = self.calc_portfolio_metric(self.trades, self.weights, "rolldown")
        self.expected_loss = self.calc_portfolio_metric(self.trades, self.weights, "expected_loss")
        self.reversion_spread = self.calc_portfolio_metric(self.trades, self.weights, "reversion_spread")
        self.break_even_spread = self.calc_portfolio_metric(self.trades, self.weights, "break_even_spread")
        self.safe_spread = self.calc_portfolio_metric(self.trades, self.weights, "safe_spread")
        self.shocked_spread = self.calc_portfolio_metric(self.trades, self.weights, "shocked_spread")
        self.worst_spread = self.calc_portfolio_metric(self.trades, self.weights, "worst_spread")
        # Calculate PnL based on the trades
        self.reversion_pnl = sum(trade.reversion_pnl for trade in self.trades)
        self.break_even_pnl = sum(trade.break_even_pnl for trade in self.trades)
        self.safe_pnl = sum(trade.safe_pnl for trade in self.trades)
        self.worst_pnl = sum(trade.worst_pnl for trade in self.trades)
        # Calculate capacity
        self.capacity = self.calc_capacity()

    def calc_weights(self) -> list[float]:
        """Calculate the weights of the trades in the portfolio."""
        total_value = sum(trade.quantity for trade in self.trades)
        if total_value == 0:
            return [0] * len(self.trades)
        return [trade.quantity / total_value for trade in self.trades]

    @staticmethod
    def calc_portfolio_metric(trades: list[Trade], weights: list[float], metric_name: str) -> float:
        return sum(getattr(trade.asset, metric_name) * weight for trade, weight in zip(trades, weights))

    def calc_capacity(self) -> float:
        pass
