from attrs import define, field
from attrs.validators import gt, lt


@define
class RiskBudget:
    return_target: float = field(validator=gt(0))
    information_ratio: float = field(validator=gt(0))
    sl_te_ratio: float = field(validator=gt(0))
    lb_te_ratio: float = field(validator=gt(0))
    tracking_error_budget: float | None = field(default=None, validator=field(gt=0))
    shock_loss_budget: float | None = field(default=None, validator=field(gt=0))
    loss_budget: float | None = field(default=None, validator=field(gt=0))

    def __attrs_post_init__(self):
        self.tracking_error_budget = self.calc_tracking_error_budget()
        self.shock_loss_budget = self.calc_shock_loss_budget()
        self.loss_budget = self.calc_loss_budget()

    def calc_tracking_error_budget(self) -> float:
        return self.return_target / self.information_ratio

    def calc_shock_loss_budget(self) -> float:
        if self.tracking_error_budget is None:
            raise ValueError("Tracking error budget must be calculated before shock loss budget.")
        return self.tracking_error_budget * self.sl_te_ratio

    def calc_loss_budget(self) -> float:
        if self.tracking_error_budget is None:
            raise ValueError("Tracking error budget must be calculated before loss budget.")
        return self.tracking_error_budget * self.lb_te_ratio


@define
class ShockLossCapacity:
    loss_budget: float = field(validator=gt(0))  # Must be greater than 0
    drawdown: float = field(validator=gt(0))  # Must be greater than 0
    shock_loss: float = field(validator=gt(0))  # Must be greater than 0
    value: float
    value_scaler: float = field(validator=[gt(0), lt(1)])  # Must be between 0 and 1
    value_adjusted: float | None = field(default=None)
    shock_loss_adjusted: float | None = field(default=None)
    shock_loss_capacity: float | None = field(default=None)

    def __attrs_post_init__(self):
        self.value_adjusted = self.scale_value()
        self.shock_loss_adjusted = self.adjust_shock_loss()
        self.shock_loss_capacity = self.calc_shock_loss_capacity()

    def scale_value(self) -> float:
        return self.value * self.value_scaler

    def adjust_shock_loss(self) -> float:
        return self.shock_loss - self.value_adjusted

    def calc_shock_loss_capacity(self) -> float:
        return self.loss_budget - self.drawdown - self.shock_loss_adjusted
