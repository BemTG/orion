use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(5);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 14209, sign: true });
    data.append(FP16x16 { mag: 119725, sign: true });
    data.append(FP16x16 { mag: 72349, sign: true });
    data.append(FP16x16 { mag: 18876, sign: false });
    data.append(FP16x16 { mag: 86727, sign: false });
    data.append(FP16x16 { mag: 52536, sign: false });
    data.append(FP16x16 { mag: 57315, sign: false });
    data.append(FP16x16 { mag: 55084, sign: false });
    data.append(FP16x16 { mag: 49528, sign: false });
    data.append(FP16x16 { mag: 14012, sign: true });
    data.append(FP16x16 { mag: 141445, sign: true });
    data.append(FP16x16 { mag: 47410, sign: true });
    data.append(FP16x16 { mag: 81307, sign: false });
    data.append(FP16x16 { mag: 139579, sign: false });
    data.append(FP16x16 { mag: 75824, sign: true });
    data.append(FP16x16 { mag: 105645, sign: false });
    data.append(FP16x16 { mag: 53041, sign: false });
    data.append(FP16x16 { mag: 2922, sign: false });
    data.append(FP16x16 { mag: 8266, sign: false });
    data.append(FP16x16 { mag: 27709, sign: false });
    data.append(FP16x16 { mag: 40424, sign: false });
    data.append(FP16x16 { mag: 59102, sign: true });
    data.append(FP16x16 { mag: 26022, sign: false });
    data.append(FP16x16 { mag: 73637, sign: true });
    data.append(FP16x16 { mag: 39565, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
