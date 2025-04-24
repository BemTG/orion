mod input_0;
mod input_1;
mod input_2;
mod input_3;
mod output_0;


use orion::operators::nn::NNTrait;
use orion::numbers::FixedTrait;
use orion::operators::nn::FP8x23NN;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::FP8x23TensorPartialEq;

#[test]
#[available_gas(2000000000)]
fn test_gru_fp8x23_with_initial_bias() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let input_2 = input_2::input_2();
    let input_3 = input_3::input_3();
    let z = output_0::output_0();

    let y = NNTrait::gru( @input_0, @input_1, @input_2, Option::Some(input_3), Option::None(()), Option::None(()), Option::None(()), Option::None(()), Option::None(()), Option::None(()), Option::None(()), Option::None(()), Option::None(()), Option::None(()), Option::Some(2)  ) ;

    assert_seq_eq(y, z);
}
