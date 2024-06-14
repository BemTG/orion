use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(15);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 71434, sign: true });
    data.append(FP16x16 { mag: 126484, sign: false });
    data.append(FP16x16 { mag: 55705, sign: true });
    data.append(FP16x16 { mag: 7208, sign: false });
    data.append(FP16x16 { mag: 7864, sign: true });
    data.append(FP16x16 { mag: 28180, sign: true });
    data.append(FP16x16 { mag: 110100, sign: false });
    data.append(FP16x16 { mag: 47841, sign: false });
    data.append(FP16x16 { mag: 53084, sign: true });
    data.append(FP16x16 { mag: 20971, sign: false });
    data.append(FP16x16 { mag: 24248, sign: false });
    data.append(FP16x16 { mag: 14417, sign: false });
    data.append(FP16x16 { mag: 9175, sign: false });
    data.append(FP16x16 { mag: 49152, sign: true });
    data.append(FP16x16 { mag: 53739, sign: false });
    data.append(FP16x16 { mag: 51773, sign: true });
    data.append(FP16x16 { mag: 5242, sign: false });
    data.append(FP16x16 { mag: 1310, sign: false });
    data.append(FP16x16 { mag: 655, sign: false });
    data.append(FP16x16 { mag: 38666, sign: true });
    data.append(FP16x16 { mag: 57671, sign: true });
    data.append(FP16x16 { mag: 26214, sign: false });
    data.append(FP16x16 { mag: 93061, sign: false });
    data.append(FP16x16 { mag: 144179, sign: false });
    data.append(FP16x16 { mag: 47841, sign: true });
    data.append(FP16x16 { mag: 2621, sign: false });
    data.append(FP16x16 { mag: 90439, sign: false });
    data.append(FP16x16 { mag: 61603, sign: false });
    data.append(FP16x16 { mag: 33423, sign: false });
    data.append(FP16x16 { mag: 165150, sign: false });
    data.append(FP16x16 { mag: 15073, sign: false });
    data.append(FP16x16 { mag: 79298, sign: false });
    data.append(FP16x16 { mag: 3276, sign: false });
    data.append(FP16x16 { mag: 16384, sign: true });
    data.append(FP16x16 { mag: 27525, sign: false });
    data.append(FP16x16 { mag: 6553, sign: false });
    data.append(FP16x16 { mag: 28180, sign: true });
    data.append(FP16x16 { mag: 13107, sign: false });
    data.append(FP16x16 { mag: 58327, sign: true });
    data.append(FP16x16 { mag: 87818, sign: true });
    data.append(FP16x16 { mag: 7208, sign: true });
    data.append(FP16x16 { mag: 74711, sign: true });
    data.append(FP16x16 { mag: 26869, sign: true });
    data.append(FP16x16 { mag: 139591, sign: false });
    data.append(FP16x16 { mag: 68157, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
