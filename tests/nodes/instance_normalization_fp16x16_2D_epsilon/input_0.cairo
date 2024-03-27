use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 24613, sign: true });
    data.append(FP16x16 { mag: 27106, sign: false });
    data.append(FP16x16 { mag: 29975, sign: true });
    data.append(FP16x16 { mag: 23640, sign: true });
    data.append(FP16x16 { mag: 38889, sign: true });
    data.append(FP16x16 { mag: 31563, sign: false });
    data.append(FP16x16 { mag: 28224, sign: true });
    data.append(FP16x16 { mag: 8679, sign: true });
    data.append(FP16x16 { mag: 87508, sign: false });
    data.append(FP16x16 { mag: 6397, sign: true });
    data.append(FP16x16 { mag: 28616, sign: true });
    data.append(FP16x16 { mag: 90574, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
