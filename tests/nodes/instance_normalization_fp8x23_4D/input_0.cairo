use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(1);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 10744192, sign: true });
    data.append(FP8x23 { mag: 8651591, sign: false });
    data.append(FP8x23 { mag: 20927410, sign: false });
    data.append(FP8x23 { mag: 3290146, sign: false });
    data.append(FP8x23 { mag: 1605848, sign: true });
    data.append(FP8x23 { mag: 7141571, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
