use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(5);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 943, sign: false });
    data.append(FP16x16 { mag: 135, sign: false });
    data.append(FP16x16 { mag: 1824, sign: false });
    data.append(FP16x16 { mag: 5933, sign: false });
    data.append(FP16x16 { mag: 12764, sign: false });
    data.append(FP16x16 { mag: 7954, sign: false });
    data.append(FP16x16 { mag: 12021, sign: false });
    data.append(FP16x16 { mag: 3426, sign: false });
    data.append(FP16x16 { mag: 9535, sign: false });
    data.append(FP16x16 { mag: 12347, sign: false });
    data.append(FP16x16 { mag: 8288, sign: false });
    data.append(FP16x16 { mag: 3404, sign: false });
    data.append(FP16x16 { mag: 10248, sign: false });
    data.append(FP16x16 { mag: 5713, sign: false });
    data.append(FP16x16 { mag: 10634, sign: false });
    data.append(FP16x16 { mag: 791, sign: false });
    data.append(FP16x16 { mag: 5695, sign: false });
    data.append(FP16x16 { mag: 569, sign: false });
    data.append(FP16x16 { mag: 12263, sign: false });
    data.append(FP16x16 { mag: 477, sign: false });
    data.append(FP16x16 { mag: 4588, sign: false });
    data.append(FP16x16 { mag: 4702, sign: false });
    data.append(FP16x16 { mag: 8308, sign: false });
    data.append(FP16x16 { mag: 1035, sign: false });
    data.append(FP16x16 { mag: 10158, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
