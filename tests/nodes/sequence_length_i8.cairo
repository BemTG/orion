mod input_0;
mod output_0;


use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::U32TensorPartialEq;
use orion::operators::sequence::U32Sequence;
use orion::operators::tensor::I8TensorPartialEq;
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::sequence::SequenceTrait;
use orion::operators::sequence::I8Sequence;

#[test]
#[available_gas(2000000000)]
fn test_sequence_length_i8() {
    let input_0 = input_0::input_0();
    let z = output_0::output_0();

    let y = input_0.sequence_length();

    assert_eq(y, z);
}
