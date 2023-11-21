mod input_0;
mod input_1;
mod input_2;
mod output_0;


use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::TensorTrait;
use orion::operators::tensor::FP8x23Tensor;
use orion::operators::tensor::FP8x23TensorPartialEq;
use orion::utils::assert_eq;

#[test]
#[available_gas(2000000000)]
fn test_scatter_fp8x23_axis1() {
    let input_0 = input_0::input_0();
    let input_1 = input_1::input_1();
    let input_2 = input_2::input_2();
    let z = output_0::output_0();

    let y = input_0
        .scatter(
            updates: input_1,
            indices: input_2,
            axis: Option::Some(1),
            reduction: Option::Some('none')
        );

    assert_eq(y, z);
}
