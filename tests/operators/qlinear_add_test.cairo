use debug::PrintTrait;
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, I8Tensor, I32Tensor, U32Tensor, FP16x16Tensor};
use orion::numbers::{FP16x16, FP16x16Impl, FP32x32, FP32x32Impl, FixedTrait};
use orion::numbers::{NumberTrait, IntegerTrait};
use orion::numbers::{i8, i32};


#[test]
#[available_gas(200000000000)]
fn qlinearadd_test() {
    let a = TensorTrait::<
        i8
    >::new(
        shape: array![4, 2].span(),
        data: array![
            IntegerTrait::<i8>::new(1_u8, false),
            IntegerTrait::<i8>::new(2_u8, false),
            IntegerTrait::<i8>::new(3_u8, false),
            IntegerTrait::<i8>::new(4_u8, false),
            IntegerTrait::<i8>::new(5_u8, false),
            IntegerTrait::<i8>::new(6_u8, false),
            IntegerTrait::<i8>::new(7_u8, false),
            IntegerTrait::<i8>::new(8_u8, false)
        ]
            .span(),
    );
    let b = TensorTrait::<
        i8
    >::new(
        shape: array![4, 2].span(),
        data: array![
            IntegerTrait::<i8>::new(2_u8, false),
            IntegerTrait::<i8>::new(4_u8, false),
            IntegerTrait::<i8>::new(6_u8, false),
            IntegerTrait::<i8>::new(8_u8, false),
            IntegerTrait::<i8>::new(10_u8, false),
            IntegerTrait::<i8>::new(12_u8, false),
            IntegerTrait::<i8>::new(14_u8, false),
            IntegerTrait::<i8>::new(16_u8, false)
        ]
            .span(),
    );

    let a_scale = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(20000, false)].span(),);
    let a_zero_point = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(0, false)].span(),);
    let b_scale = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(25000, false)].span(),);
    let b_zero_point = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(0, false)].span(),);

    let y_scale = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(30000, false)].span(),);
    let y_zero_point = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(0, false)].span(),);

    let actual_output = a
        .qlinear_add(@a_scale, @a_zero_point, @b, @b_scale, @b_zero_point, @y_scale, @y_zero_point);

    assert((*actual_output.data[0]).into() == 2, '*result[0] == 2');
    assert((*actual_output.data[1]).into() == 4, '*result[1] == 4');
    assert((*actual_output.data[2]).into() == 7, '*result[2] == 7');
    assert((*actual_output.data[3]).into() == 9, '*result[3] == 9');
    assert((*actual_output.data[4]).into() == 11, '*result[4] == 11');
    assert((*actual_output.data[5]).into() == 14, '*result[5] == 14');
    assert((*actual_output.data[6]).into() == 16, '*result[6] == 16');
    assert((*actual_output.data[7]).into() == 18, '*result[7] == 18');
}

#[test]
#[available_gas(200000000000)]
fn qlinearadd_broadcast_test() {
    let a = TensorTrait::<
        i8
    >::new(
        shape: array![2, 4].span(),
        data: array![
            IntegerTrait::<i8>::new(1_u8, false),
            IntegerTrait::<i8>::new(2_u8, false),
            IntegerTrait::<i8>::new(3_u8, false),
            IntegerTrait::<i8>::new(4_u8, false),
            IntegerTrait::<i8>::new(5_u8, false),
            IntegerTrait::<i8>::new(6_u8, false),
            IntegerTrait::<i8>::new(7_u8, false),
            IntegerTrait::<i8>::new(8_u8, false)
        ]
            .span(),
    );
    let b = TensorTrait::<
        i8
    >::new(
        shape: array![1, 4].span(),
        data: array![
            IntegerTrait::<i8>::new(2_u8, false),
            IntegerTrait::<i8>::new(4_u8, false),
            IntegerTrait::<i8>::new(6_u8, false),
            IntegerTrait::<i8>::new(8_u8, false),
        ]
            .span(),
    );

    let a_scale = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(20000, false)].span(),);
    let a_zero_point = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(0, false)].span(),);
    let b_scale = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(25000, false)].span(),);
    let b_zero_point = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(0, false)].span(),);

    let y_scale = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(30000, false)].span(),);
    let y_zero_point = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(0, false)].span(),);

    let actual_output = a
        .qlinear_add(@a_scale, @a_zero_point, @b, @b_scale, @b_zero_point, @y_scale, @y_zero_point);

    assert((*actual_output.data[0]).into() == 2, '*result[0] == 2');
    assert((*actual_output.data[1]).into() == 4, '*result[1] == 4');
    assert((*actual_output.data[2]).into() == 7, '*result[2] == 7');
    assert((*actual_output.data[3]).into() == 9, '*result[3] == 9');
    assert((*actual_output.data[4]).into() == 5, '*result[4] == 5');
    assert((*actual_output.data[5]).into() == 7, '*result[5] == 7');
    assert((*actual_output.data[6]).into() == 9, '*result[6] == 9');
    assert((*actual_output.data[7]).into() == 12, '*result[7] == 12');
}


#[test]
#[available_gas(200000000000)]
fn test_example_doc() {
    let a = TensorTrait::<
        i8
    >::new(
        shape: array![2, 3].span(),
        data: array![
            IntegerTrait::<i8>::new(6_u8, false),
            IntegerTrait::<i8>::new(6_u8, false),
            IntegerTrait::<i8>::new(6_u8, false),
            IntegerTrait::<i8>::new(11_u8, false),
            IntegerTrait::<i8>::new(11_u8, false),
            IntegerTrait::<i8>::new(11_u8, false)
        ]
            .span(),
    );
    let b = TensorTrait::<
        i8
    >::new(
        shape: array![1, 3].span(),
        data: array![
            IntegerTrait::<i8>::new(40_u8, false),
            IntegerTrait::<i8>::new(40_u8, false),
            IntegerTrait::<i8>::new(40_u8, false)
        ]
            .span(),
    );

    let a_scale = TensorTrait::<
        FP16x16
    >::new(
        shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(131072, false)].span(),
    );
    let a_zero_point = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(65536, false)].span(),);
    let b_scale = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(16384, false)].span(),);
    let b_zero_point = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(0, false)].span(),);

    let y_scale = TensorTrait::<
        FP16x16
    >::new(
        shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(655360, false)].span(),
    );
    let y_zero_point = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(65536, true)].span(),);

    let actual_output = a
        .qlinear_add(@a_scale, @a_zero_point, @b, @b_scale, @b_zero_point, @y_scale, @y_zero_point);
    assert((*actual_output.data[0]).into() == 1, '*result[0] == 1');
    assert((*actual_output.data[1]).into() == 1, '*result[1] == 1');
    assert((*actual_output.data[2]).into() == 1, '*result[2] == 1');
    assert((*actual_output.data[3]).into() == 2, '*result[3] == 2');
    assert((*actual_output.data[4]).into() == 2, '*result[4] == 2');
    assert((*actual_output.data[5]).into() == 2, '*result[5] == 2');
}
