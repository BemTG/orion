mod input_0;
mod input_1;
mod input_2;
mod output_0;


use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::nn::NNTrait;
use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::numbers::FixedTrait;
use orion::operators::nn::FP8x23NN;

#[test]
#[available_gas(2000000000)]
fn test_linear_fp8x23() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let input_2 = input_2::input_2();
    let z = output_0::output_0();

    let y = NNTrait::linear(input_0, input_1, input_2);

    assert_eq(y, z);
}
