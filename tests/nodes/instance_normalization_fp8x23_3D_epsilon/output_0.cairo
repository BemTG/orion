use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 4400861, sign: true });
    data.append(FP8x23 { mag: 2613383, sign: false });
    data.append(FP8x23 { mag: 26927, sign: false });
    data.append(FP8x23 { mag: 2738105, sign: true });
    data.append(FP8x23 { mag: 5325595, sign: true });
    data.append(FP8x23 { mag: 3699882, sign: true });
    data.append(FP8x23 { mag: 4830248, sign: true });
    data.append(FP8x23 { mag: 4982918, sign: true });
    data.append(FP8x23 { mag: 537172, sign: false });
    data.append(FP8x23 { mag: 7392802, sign: false });
    data.append(FP8x23 { mag: 3078484, sign: false });
    data.append(FP8x23 { mag: 5715904, sign: false });
    data.append(FP8x23 { mag: 3135122, sign: true });
    data.append(FP8x23 { mag: 2562241, sign: true });
    data.append(FP8x23 { mag: 2157699, sign: true });
    data.append(FP8x23 { mag: 3356406, sign: false });
    data.append(FP8x23 { mag: 4130154, sign: true });
    data.append(FP8x23 { mag: 4664839, sign: true });
    data.append(FP8x23 { mag: 5486699, sign: true });
    data.append(FP8x23 { mag: 4556950, sign: true });
    data.append(FP8x23 { mag: 6593354, sign: false });
    data.append(FP8x23 { mag: 749390, sign: false });
    data.append(FP8x23 { mag: 2585910, sign: false });
    data.append(FP8x23 { mag: 6795708, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
