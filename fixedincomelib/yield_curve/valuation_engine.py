"""Analytic valuation of a signed fixed-accrual cashflow on an IFR model."""

import numpy as np

from fixedincomelib.model import ModelType
from fixedincomelib.product import ProductFixedAccruedCashflow
from fixedincomelib.valuation import (
    AnalyticValParam, FundingIndexParameter, ValuationEngineProduct,
    ValuationEngineProductRegistry,
)


class ValuationEngineProductFixedAccrued(ValuationEngineProduct):
    def __init__(self, model, valuation_parameters_collection, product, request):
        super().__init__(model, valuation_parameters_collection, product, request)
        # TODO 
        funding_parameter = (
            valuation_parameters_collection
            .get_vp_from_build_method_collection(
                FundingIndexParameter._vp_type
            )
        )

        self.funding_index_ = funding_parameter.get_funding_index(
            product.currency
        )

    def calculate_value(self) -> None:
        # TODO Set self.value_ and self.cash_ for past, same-day, and future payments.
        self.value_ = 0.0
        self.cash_ = 0.0

        payment_date = self.product_.payment_date
        cash_amount = self.product_.notional * self.product_.accrued

        if payment_date < self.value_date:
            return

        if payment_date == self.value_date:
            self.value_ = cash_amount
            self.cash_ = cash_amount
        else:
            discount_factor = self.model.discount_factor(
                self.funding_index_, payment_date
            )
            self.value_ = cash_amount * discount_factor

    def calculate_risk(
        self, gradient: list[np.ndarray], scaler: float = 1.0, accumulate: bool = False,
    ) -> None:
        # TODO Write product risk using the model's DF gradient; non-future payments have zero risk.
        # Apply scaler to the new contribution; overwrite or add according to accumulate.
        self.model.resize_gradient(gradient)

        if not accumulate:
            for block in gradient:
                block.fill(0.0)

        payment_date = self.product_.payment_date

        if payment_date <= self.value_date:
            return

        cash_amount = self.product_.notional * self.product_.accrued

        self.model.discount_factor_gradient_wrt_state(
            self.funding_index_,
            payment_date,
            gradient,
            scaler=scaler * cash_amount,
            accumulate=True,
        )


ValuationEngineProductRegistry().register(
    (ModelType.YIELD_CURVE.to_string(), ProductFixedAccruedCashflow._product_type, AnalyticValParam._vp_type),
    ValuationEngineProductFixedAccrued,
)
