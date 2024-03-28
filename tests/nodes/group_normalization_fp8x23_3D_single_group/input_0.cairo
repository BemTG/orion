use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 6830677, sign: true });
    data.append(FP8x23 { mag: 7413847, sign: true });
    data.append(FP8x23 { mag: 4976428, sign: false });
    data.append(FP8x23 { mag: 170678, sign: false });
    data.append(FP8x23 { mag: 6098943, sign: false });
    data.append(FP8x23 { mag: 97288, sign: true });
    data.append(FP8x23 { mag: 2645269, sign: false });
    data.append(FP8x23 { mag: 12809262, sign: false });
    data.append(FP8x23 { mag: 2942631, sign: false });
    data.append(FP8x23 { mag: 6062907, sign: true });
    data.append(FP8x23 { mag: 1330458, sign: false });
    data.append(FP8x23 { mag: 29542256, sign: true });
    data.append(FP8x23 { mag: 2384915, sign: false });
    data.append(FP8x23 { mag: 7877089, sign: false });
    data.append(FP8x23 { mag: 2232837, sign: true });
    data.append(FP8x23 { mag: 1112683, sign: false });
    data.append(FP8x23 { mag: 3130651, sign: false });
    data.append(FP8x23 { mag: 12009019, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
