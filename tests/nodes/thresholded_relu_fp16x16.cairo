mod input_0;
mod output_0;


use orion::numbers::FixedTrait;
use orion::operators::nn::FP16x16NN;
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::nn::NNTrait;

#[test]
#[available_gas(2000000000)]
fn test_thresholded_relu_fp16x16() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = NNTrait::thresholded_relu(@input_0, @FixedTrait::new(65536, false));

    assert_eq(y, z);
}
