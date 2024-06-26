mod input_0;
mod output_0;


use orion::operators::tensor::I32TensorPartialEq;
use orion::operators::tensor::{TensorTrait, Tensor};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{I32Tensor, I32TensorAdd};
use orion::utils::{assert_eq, assert_seq_eq};

#[test]
#[available_gas(2000000000)]
fn test_cumsum_i32_1d_exclusive() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.cumsum(0, Option::Some(true), Option::Some(false));

    assert_eq(y, z);
}
