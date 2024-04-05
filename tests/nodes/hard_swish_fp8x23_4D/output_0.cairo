use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 8725650, sign: false });
    data.append(FP8x23 { mag: 13072564, sign: false });
    data.append(FP8x23 { mag: 2347695, sign: true });
    data.append(FP8x23 { mag: 6677248, sign: false });
    data.append(FP8x23 { mag: 10559951, sign: false });
    data.append(FP8x23 { mag: 1203766, sign: true });
    data.append(FP8x23 { mag: 15059581, sign: false });
    data.append(FP8x23 { mag: 1009423, sign: true });
    data.append(FP8x23 { mag: 2984988, sign: true });
    data.append(FP8x23 { mag: 2658678, sign: true });
    data.append(FP8x23 { mag: 701193, sign: true });
    data.append(FP8x23 { mag: 9581028, sign: false });
    data.append(FP8x23 { mag: 21689796, sign: false });
    data.append(FP8x23 { mag: 4186971, sign: false });
    data.append(FP8x23 { mag: 2606874, sign: true });
    data.append(FP8x23 { mag: 3139280, sign: true });
    data.append(FP8x23 { mag: 24763098, sign: false });
    data.append(FP8x23 { mag: 2506124, sign: true });
    data.append(FP8x23 { mag: 18846234, sign: false });
    data.append(FP8x23 { mag: 2990722, sign: true });
    data.append(FP8x23 { mag: 717194, sign: true });
    data.append(FP8x23 { mag: 1125726, sign: true });
    data.append(FP8x23 { mag: 613343, sign: false });
    data.append(FP8x23 { mag: 15232088, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
