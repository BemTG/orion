use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 5070906, sign: false });
    data.append(FP8x23 { mag: 2007974, sign: true });
    data.append(FP8x23 { mag: 8550920, sign: true });
    data.append(FP8x23 { mag: 5317807, sign: false });
    data.append(FP8x23 { mag: 5524318, sign: false });
    data.append(FP8x23 { mag: 4507204, sign: true });
    data.append(FP8x23 { mag: 1545045, sign: false });
    data.append(FP8x23 { mag: 10910858, sign: false });
    data.append(FP8x23 { mag: 3606904, sign: false });
    data.append(FP8x23 { mag: 9167679, sign: false });
    data.append(FP8x23 { mag: 12550389, sign: false });
    data.append(FP8x23 { mag: 13395357, sign: false });
    data.append(FP8x23 { mag: 2126579, sign: true });
    data.append(FP8x23 { mag: 9128824, sign: false });
    data.append(FP8x23 { mag: 4209100, sign: false });
    data.append(FP8x23 { mag: 16715701, sign: true });
    data.append(FP8x23 { mag: 880403, sign: false });
    data.append(FP8x23 { mag: 7460805, sign: false });
    data.append(FP8x23 { mag: 5701920, sign: false });
    data.append(FP8x23 { mag: 4419670, sign: true });
    data.append(FP8x23 { mag: 1604894, sign: false });
    data.append(FP8x23 { mag: 8424695, sign: true });
    data.append(FP8x23 { mag: 1140199, sign: false });
    data.append(FP8x23 { mag: 365439, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
