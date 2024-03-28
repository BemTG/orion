use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 21767854, sign: false });
    data.append(FP8x23 { mag: 3771696, sign: false });
    data.append(FP8x23 { mag: 3869813, sign: false });
    data.append(FP8x23 { mag: 1256359, sign: false });
    data.append(FP8x23 { mag: 16294185, sign: true });
    data.append(FP8x23 { mag: 1354291, sign: false });
    data.append(FP8x23 { mag: 21105694, sign: false });
    data.append(FP8x23 { mag: 10365678, sign: true });
    data.append(FP8x23 { mag: 5543294, sign: true });
    data.append(FP8x23 { mag: 11251028, sign: false });
    data.append(FP8x23 { mag: 4513084, sign: true });
    data.append(FP8x23 { mag: 4216892, sign: false });
    data.append(FP8x23 { mag: 15558267, sign: true });
    data.append(FP8x23 { mag: 4857629, sign: false });
    data.append(FP8x23 { mag: 5918408, sign: true });
    data.append(FP8x23 { mag: 7598979, sign: true });
    data.append(FP8x23 { mag: 3224546, sign: false });
    data.append(FP8x23 { mag: 378339, sign: true });
    data.append(FP8x23 { mag: 6620033, sign: false });
    data.append(FP8x23 { mag: 2158059, sign: true });
    data.append(FP8x23 { mag: 8875864, sign: true });
    data.append(FP8x23 { mag: 235100, sign: false });
    data.append(FP8x23 { mag: 8570879, sign: false });
    data.append(FP8x23 { mag: 15295647, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
