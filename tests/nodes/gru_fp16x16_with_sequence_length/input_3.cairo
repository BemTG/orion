use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_3() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(30);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 51773, sign: false });
    data.append(FP16x16 { mag: 69468, sign: true });
    data.append(FP16x16 { mag: 9175, sign: false });
    data.append(FP16x16 { mag: 11796, sign: true });
    data.append(FP16x16 { mag: 95027, sign: false });
    data.append(FP16x16 { mag: 5898, sign: false });
    data.append(FP16x16 { mag: 66191, sign: true });
    data.append(FP16x16 { mag: 58327, sign: true });
    data.append(FP16x16 { mag: 18350, sign: false });
    data.append(FP16x16 { mag: 63569, sign: true });
    data.append(FP16x16 { mag: 28835, sign: true });
    data.append(FP16x16 { mag: 117309, sign: true });
    data.append(FP16x16 { mag: 49807, sign: false });
    data.append(FP16x16 { mag: 92405, sign: false });
    data.append(FP16x16 { mag: 125173, sign: true });
    data.append(FP16x16 { mag: 7208, sign: true });
    data.append(FP16x16 { mag: 3276, sign: false });
    data.append(FP16x16 { mag: 25559, sign: true });
    data.append(FP16x16 { mag: 34734, sign: false });
    data.append(FP16x16 { mag: 9830, sign: false });
    data.append(FP16x16 { mag: 13107, sign: false });
    data.append(FP16x16 { mag: 71434, sign: false });
    data.append(FP16x16 { mag: 3276, sign: true });
    data.append(FP16x16 { mag: 45219, sign: false });
    data.append(FP16x16 { mag: 57671, sign: true });
    data.append(FP16x16 { mag: 100270, sign: true });
    data.append(FP16x16 { mag: 7864, sign: true });
    data.append(FP16x16 { mag: 54394, sign: false });
    data.append(FP16x16 { mag: 15728, sign: true });
    data.append(FP16x16 { mag: 34078, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
