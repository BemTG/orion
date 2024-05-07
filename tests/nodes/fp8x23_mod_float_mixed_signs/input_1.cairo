use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 30229106, sign: false });
    data.append(FP8x23 { mag: 46594112, sign: false });
    data.append(FP8x23 { mag: 41459500, sign: false });
    data.append(FP8x23 { mag: 9129659, sign: false });
    data.append(FP8x23 { mag: 39546836, sign: false });
    data.append(FP8x23 { mag: 78909664, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
