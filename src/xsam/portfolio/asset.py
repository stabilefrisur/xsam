from attrs import define, field
import pandas as pd


@define
class Asset:
    data: pd.DataFrame | None = field(default=None)
    as_of_date: pd.Timestamp | None = field(default=None)
    # Modelled return and risk
    convex: bool = field(default=False)
    carry: float | None = field(default=None)
    value: float | None = field(default=None)
    spread_shock: float | None = field(default=None)
    shock_loss: float | None = field(default=None)
    beta: float | None = field(default=None)
    ctp: float | None = field(default=None)
    # Market data
    price: float | None = field(default=None)
    spread: float | None = field(default=None)
    rolldown: float | None = field(default=None)
    expected_loss: float | None = field(default=None)
    spread_duration: float | None = field(default=None)
    # Calculated spreads
    reversion_spread: float | None = field(default=None)
    break_even_spread: float | None = field(default=None)
    safe_spread: float | None = field(default=None)
    shocked_spread: float | None = field(default=None)
    worst_spread: float | None = field(default=None)
    # Calculated prices
    reversion_price: float | None = field(default=None)
    break_even_price: float | None = field(default=None)
    safe_price: float | None = field(default=None)
    shocked_price: float | None = field(default=None)
    worst_price: float | None = field(default=None)
    # Calculated returns
    reversion_ret: float | None = field(default=None)
    break_even_ret: float | None = field(default=None)
    shocked_ret: float | None = field(default=None)
    safe_ret: float | None = field(default=None)
    worst_ret: float | None = field(default=None)

    # Mapping of DataFrame column names to attribute names
    COLUMN_TO_ATTR_MAP = {
        "Carry": "carry",
        "Value": "value",
        "Spread Shock": "spread_shock",
        "Shock Loss": "shock_loss",
        "Beta": "beta",
        "CTP": "ctp",
        "Price - Cvx": "price",
        "Spread Duration": "spread_duration",
        "Spread": "spread",
        "Rolldown": "rolldown",
        "Expected Loss": "expected_loss",
    }

    def __attrs_post_init__(self):
        if self.data is not None:
            # Ensure the DataFrame index is a DatetimeIndex
            if not isinstance(self.data.index, pd.DatetimeIndex):
                raise ValueError("The index of data must be a DatetimeIndex.")

            # Select the row based on the date or use the last row
            if self.as_of_date is not None and self.as_of_date in self.data.index:
                row_to_use = self.data.loc[self.as_of_date]
            else:
                row_to_use = self.data.iloc[-1]

            # Populate attributes from the selected row
            for column, attr in self.COLUMN_TO_ATTR_MAP.items():
                # Only update the attribute if it wasn't explicitly provided
                if getattr(self, attr) is None and column in row_to_use:
                    setattr(self, attr, row_to_use[column])

        self.reversion_spread = self.calc_reversion_spread()
        self.break_even_spread = self.calc_break_even_spread()
        self.shocked_spread = self.calc_shocked_spread()
        self.safe_spread = self.calc_safe_spread()
        self.worst_spread = self.extract_worst_spread()
        self.reversion_price = self.calc_reversion_price()
        self.break_even_price = self.calc_break_even_price()
        self.shocked_price = self.calc_shocked_price()
        self.safe_price = self.calc_safe_price()
        self.worst_price = self.extract_worst_price()
        self.reversion_ret = self.calc_reversion_ret_price() if self.convex else self.calc_reversion_ret_spread()
        self.break_even_ret = self.calc_break_even_ret_price() if self.convex else self.calc_break_even_ret_spread()
        self.shocked_ret = self.calc_shocked_ret_price() if self.convex else self.calc_shocked_ret_spread()
        self.safe_ret = self.calc_safe_ret_price() if self.convex else self.calc_safe_ret_spread()
        self.worst_ret = self.calc_worst_ret_price() if self.convex else self.calc_worst_ret_spread()

    # Calculate critical spread levels
    def calc_reversion_spread(self) -> float:
        return self.spread - self.value / self.spread_duration

    def calc_break_even_spread(self) -> float:
        return self.spread + self.carry / self.spread_duration

    def calc_shocked_spread(self) -> float:
        return self.reversion_spread + self.spread_shock

    def calc_safe_spread(self) -> float:
        return (self.shocked_spread * self.spread_duration - self.rolldown * (1 - self.expected_loss)) / (
            self.spread_duration + 1 - self.expected_loss
        )

    def extract_worst_spread(self) -> float:
        return self.data["Spread"].max()

    # Calculate critical price levels
    def calc_reversion_price(self) -> float:
        return self.price * (1 + self.value / 1e4)

    def calc_break_even_price(self) -> float:
        return self.price * (1 - self.carry / 1e4)

    def calc_shocked_price(self) -> float:
        return self.reversion_price * (1 - self.shock_loss / 1e4)

    def calc_safe_price(self) -> float:
        return self.shocked_price * (1 + self.break_even_spread * (1 - self.expected_loss) / 1e4)

    def extract_worst_price(self) -> float:
        return self.data["Price - Cvx"].min()

    # Calculate returns based on spread changes
    @staticmethod
    def calc_ret_spread(spread_0, spread_1, spread_duration):
        """Calculate the return based on spread change and spread duration."""
        return -spread_duration * (spread_1 - spread_0)

    def calc_reversion_ret_spread(self) -> float:
        return self.calc_ret_spread(self.spread, self.reversion_spread, self.spread_duration)

    def calc_break_even_ret_spread(self) -> float:
        return self.calc_ret_spread(self.spread, self.break_even_spread, self.spread_duration)

    def calc_shocked_ret_spread(self) -> float:
        return self.calc_ret_spread(self.spread, self.shocked_spread, self.spread_duration)

    def calc_safe_ret_spread(self) -> float:
        return self.calc_ret_spread(self.spread, self.safe_spread, self.spread_duration)

    def calc_worst_ret_spread(self) -> float:
        return self.calc_ret_spread(self.spread, self.worst_spread, self.spread_duration)

    # Calculate returns based on price changes
    @staticmethod
    def calc_ret_price(price_0, price_1):
        """Calculate the return based on price change."""
        return (price_1 - price_0) / price_0 * 1e4

    def calc_reversion_ret_price(self) -> float:
        return self.calc_ret_price(self.price, self.reversion_price)

    def calc_break_even_ret_price(self) -> float:
        return self.calc_ret_price(self.price, self.break_even_price)

    def calc_shocked_ret_price(self) -> float:
        return self.calc_ret_price(self.price, self.shocked_price)

    def calc_safe_ret_price(self) -> float:
        return self.calc_ret_price(self.price, self.safe_price)

    def calc_worst_ret_price(self) -> float:
        return self.calc_ret_price(self.price, self.worst_price)
