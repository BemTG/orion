mod input_0;
mod output_0;


use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::operators::tensor::{TensorTrait, Tensor};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};

#[test]
#[available_gas(2000000000)]
fn test_reduce_sum_single_axis_fp16x16_2D_axis_1() {
    let input_0 = input_0::input_0();
    let z_0 = output_0::output_0();

    let y_0 = input_0.reduce_sum_single_axis(1, false);

    assert_eq(y_0, z_0);
}