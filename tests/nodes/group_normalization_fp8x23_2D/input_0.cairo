use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 8959610, sign: true });
    data.append(FP8x23 { mag: 3719077, sign: false });
    data.append(FP8x23 { mag: 1001560, sign: false });
    data.append(FP8x23 { mag: 7156217, sign: false });
    data.append(FP8x23 { mag: 12882644, sign: false });
    data.append(FP8x23 { mag: 3059705, sign: false });
    data.append(FP8x23 { mag: 5876195, sign: false });
    data.append(FP8x23 { mag: 3853638, sign: false });
    data.append(FP8x23 { mag: 387741, sign: false });
    data.append(FP8x23 { mag: 2441033, sign: true });
    data.append(FP8x23 { mag: 4887267, sign: false });
    data.append(FP8x23 { mag: 10839239, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
