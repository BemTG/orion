use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 8014354, sign: false });
    data.append(FP8x23 { mag: 17084226, sign: false });
    data.append(FP8x23 { mag: 18422088, sign: true });
    data.append(FP8x23 { mag: 16014801, sign: true });
    data.append(FP8x23 { mag: 14359224, sign: false });
    data.append(FP8x23 { mag: 19924674, sign: false });
    data.append(FP8x23 { mag: 7773912, sign: false });
    data.append(FP8x23 { mag: 21511244, sign: false });
    data.append(FP8x23 { mag: 7070900, sign: true });
    data.append(FP8x23 { mag: 17877856, sign: true });
    data.append(FP8x23 { mag: 18557870, sign: false });
    data.append(FP8x23 { mag: 7368394, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
