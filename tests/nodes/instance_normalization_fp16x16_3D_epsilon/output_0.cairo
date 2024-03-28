use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(5);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 42036, sign: false });
    data.append(FP16x16 { mag: 64953, sign: false });
    data.append(FP16x16 { mag: 56608, sign: false });
    data.append(FP16x16 { mag: 54460, sign: false });
    data.append(FP16x16 { mag: 30927, sign: true });
    data.append(FP16x16 { mag: 16878, sign: true });
    data.append(FP16x16 { mag: 45450, sign: true });
    data.append(FP16x16 { mag: 76212, sign: false });
    data.append(FP16x16 { mag: 76070, sign: false });
    data.append(FP16x16 { mag: 73693, sign: false });
    data.append(FP16x16 { mag: 81775, sign: false });
    data.append(FP16x16 { mag: 70127, sign: false });
    data.append(FP16x16 { mag: 119183, sign: false });
    data.append(FP16x16 { mag: 110746, sign: false });
    data.append(FP16x16 { mag: 75541, sign: false });
    data.append(FP16x16 { mag: 164771, sign: false });
    data.append(FP16x16 { mag: 89122, sign: false });
    data.append(FP16x16 { mag: 42946, sign: false });
    data.append(FP16x16 { mag: 64852, sign: false });
    data.append(FP16x16 { mag: 17953, sign: true });
    data.append(FP16x16 { mag: 37940, sign: false });
    data.append(FP16x16 { mag: 43394, sign: false });
    data.append(FP16x16 { mag: 87577, sign: false });
    data.append(FP16x16 { mag: 49147, sign: false });
    data.append(FP16x16 { mag: 27249, sign: true });
    data.append(FP16x16 { mag: 119218, sign: false });
    data.append(FP16x16 { mag: 32927, sign: true });
    data.append(FP16x16 { mag: 76086, sign: true });
    data.append(FP16x16 { mag: 82288, sign: false });
    data.append(FP16x16 { mag: 76427, sign: false });
    data.append(FP16x16 { mag: 72920, sign: false });
    data.append(FP16x16 { mag: 70029, sign: false });
    data.append(FP16x16 { mag: 116018, sign: false });
    data.append(FP16x16 { mag: 109041, sign: false });
    data.append(FP16x16 { mag: 76962, sign: false });
    data.append(FP16x16 { mag: 168221, sign: false });
    data.append(FP16x16 { mag: 51191, sign: false });
    data.append(FP16x16 { mag: 94534, sign: false });
    data.append(FP16x16 { mag: 8094, sign: true });
    data.append(FP16x16 { mag: 41336, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
