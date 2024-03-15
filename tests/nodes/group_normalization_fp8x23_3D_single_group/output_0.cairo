use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 515347, sign: false });
    data.append(FP8x23 { mag: 19577184, sign: true });
    data.append(FP8x23 { mag: 2204489, sign: true });
    data.append(FP8x23 { mag: 645655, sign: false });
    data.append(FP8x23 { mag: 737294, sign: true });
    data.append(FP8x23 { mag: 9565351, sign: true });
    data.append(FP8x23 { mag: 662194, sign: true });
    data.append(FP8x23 { mag: 17754340, sign: true });
    data.append(FP8x23 { mag: 2731513, sign: true });
    data.append(FP8x23 { mag: 144955, sign: true });
    data.append(FP8x23 { mag: 3778544, sign: false });
    data.append(FP8x23 { mag: 19728108, sign: true });
    data.append(FP8x23 { mag: 16916230, sign: true });
    data.append(FP8x23 { mag: 17008012, sign: true });
    data.append(FP8x23 { mag: 4478172, sign: true });
    data.append(FP8x23 { mag: 2057753, sign: false });
    data.append(FP8x23 { mag: 3143511, sign: false });
    data.append(FP8x23 { mag: 13241452, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
