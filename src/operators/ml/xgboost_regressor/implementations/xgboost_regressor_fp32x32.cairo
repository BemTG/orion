use orion::operators::ml::xgboost_regressor::core::{TreeRegressor, XGBoostRegressorTrait, predict};
use orion::operators::ml::xgboost_regressor::core;
use orion::operators::ml::FP32x32TreeRegressor;
use orion::numbers::{FP32x32, FP32x32Impl};

impl FP32x32XGBoostRegressor of XGBoostRegressorTrait<FP32x32> {
    fn predict(
        ref self: Span<TreeRegressor<FP32x32>>,
        ref features: Span<FP32x32>,
        ref weights: Span<FP32x32>
    ) -> FP32x32 {
        predict(ref self, ref features, ref weights)
    }
}
