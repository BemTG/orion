mod input_0;
mod input_1;
mod input_2;
mod output_0;


use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::operators::nn::FP8x23NN;
use orion::operators::nn::NNTrait;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::numbers::FixedTrait;

#[test]
#[available_gas(2000000000)]
fn test_rnn_fp8x23_with_batchwise() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let input_2 = input_2::input_2();
    let z = output_0::output_0();

    let y = NNTrait::rnn( @input_0, @input_1, @input_2, Option::None(()), Option::None(()), Option::None(()), Option::None(()), Option::None(()), Option::None(()), Option::None(()), Option::None(()), Option::None(()), Option::Some(1), Option::None(()), Option::Some(2)  ) ;

    assert_seq_eq(y, z);
}
