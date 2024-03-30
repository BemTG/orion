use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 71614, sign: false });
    data.append(FP16x16 { mag: 8744, sign: false });
    data.append(FP16x16 { mag: 98761, sign: true });
    data.append(FP16x16 { mag: 80223, sign: false });
    data.append(FP16x16 { mag: 69150, sign: true });
    data.append(FP16x16 { mag: 29517, sign: false });
    data.append(FP16x16 { mag: 64476, sign: true });
    data.append(FP16x16 { mag: 161321, sign: false });
    data.append(FP16x16 { mag: 10609, sign: true });
    data.append(FP16x16 { mag: 27815, sign: true });
    data.append(FP16x16 { mag: 44851, sign: true });
    data.append(FP16x16 { mag: 64075, sign: false });
    data.append(FP16x16 { mag: 3771, sign: false });
    data.append(FP16x16 { mag: 5594, sign: false });
    data.append(FP16x16 { mag: 30515, sign: true });
    data.append(FP16x16 { mag: 50317, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
