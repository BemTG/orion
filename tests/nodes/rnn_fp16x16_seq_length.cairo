mod input_0;
mod input_1;
mod input_2;
mod input_3;
mod output_0;


use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::nn::FP16x16NN;
use orion::operators::nn::NNTrait;
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::numbers::FixedTrait;

#[test]
#[available_gas(2000000000)]
fn test_rnn_fp16x16_seq_length() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let input_2 = input_2::input_2();
    let input_3 = input_3::input_3();
    let z = output_0::output_0();

    let y = NNTrait::rnn( @input_0, @input_1, @input_2, Option::Some(input_3), Option::None(()), Option::None(()), Option::None(()), Option::None(()), Option::None(()), Option::None(()), Option::None(()), Option::None(()), Option::None(()), Option::None(()), Option::Some(2)  ) ;

    assert_seq_eq(y, z);
}
