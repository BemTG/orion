use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 53363, sign: true });
    data.append(FP16x16 { mag: 35718, sign: true });
    data.append(FP16x16 { mag: 95177, sign: false });
    data.append(FP16x16 { mag: 72770, sign: false });
    data.append(FP16x16 { mag: 81539, sign: true });
    data.append(FP16x16 { mag: 48386, sign: false });
    data.append(FP16x16 { mag: 30727, sign: true });
    data.append(FP16x16 { mag: 1684, sign: false });
    data.append(FP16x16 { mag: 9010, sign: true });
    data.append(FP16x16 { mag: 147891, sign: false });
    data.append(FP16x16 { mag: 85632, sign: true });
    data.append(FP16x16 { mag: 15382, sign: false });
    data.append(FP16x16 { mag: 26747, sign: true });
    data.append(FP16x16 { mag: 57669, sign: true });
    data.append(FP16x16 { mag: 103050, sign: true });
    data.append(FP16x16 { mag: 42112, sign: false });
    data.append(FP16x16 { mag: 37587, sign: false });
    data.append(FP16x16 { mag: 90967, sign: true });
    data.append(FP16x16 { mag: 16582, sign: false });
    data.append(FP16x16 { mag: 57176, sign: true });
    data.append(FP16x16 { mag: 50420, sign: true });
    data.append(FP16x16 { mag: 77021, sign: false });
    data.append(FP16x16 { mag: 49382, sign: true });
    data.append(FP16x16 { mag: 59931, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
