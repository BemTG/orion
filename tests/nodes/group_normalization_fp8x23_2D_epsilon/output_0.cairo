use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 9564050, sign: false });
    data.append(FP8x23 { mag: 9431540, sign: false });
    data.append(FP8x23 { mag: 3025735, sign: true });
    data.append(FP8x23 { mag: 17275438, sign: true });
    data.append(FP8x23 { mag: 14499099, sign: false });
    data.append(FP8x23 { mag: 1327136, sign: false });
    data.append(FP8x23 { mag: 2138334, sign: true });
    data.append(FP8x23 { mag: 17437608, sign: true });
    data.append(FP8x23 { mag: 14936738, sign: false });
    data.append(FP8x23 { mag: 608439, sign: false });
    data.append(FP8x23 { mag: 2926417, sign: true });
    data.append(FP8x23 { mag: 17293588, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
