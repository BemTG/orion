mod input_0;
mod input_1;
mod input_2;
mod input_3;
mod input_4;
mod input_5;
mod input_6;
mod output_0;


use orion::operators::nn::FP8x23NN;
use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::operators::nn::NNTrait;
use orion::numbers::FixedTrait;
use orion::utils::{assert_eq, assert_seq_eq};

#[test]
#[available_gas(2000000000)]
fn test_lstm_FP8x23_with_peepholes() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let input_2 = input_2::input_2();
    let input_3 = input_3::input_3();
    let input_4 = input_4::input_4();
    let input_5 = input_5::input_5();
    let input_6 = input_6::input_6();
    let z = output_0::output_0();

    let y = NNTrait::lstm( @input_0, @input_1, @input_2, Option::Some(input_3), Option::None(()), Option::Some(input_4), Option::Some(input_5), Option::Some(input_6), Option::None(()), Option::None(()), Option::None(()), Option::None(()), Option::None(()), Option::None(()), Option::None(()), Option::None(()), Option::None(()), Option::Some(2)  ) ;

    assert_seq_eq(y, z);
}
