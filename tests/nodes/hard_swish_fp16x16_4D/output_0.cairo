use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 10503, sign: false });
    data.append(FP16x16 { mag: 24218, sign: true });
    data.append(FP16x16 { mag: 21186, sign: true });
    data.append(FP16x16 { mag: 17248, sign: true });
    data.append(FP16x16 { mag: 135308, sign: false });
    data.append(FP16x16 { mag: 10959, sign: true });
    data.append(FP16x16 { mag: 19348, sign: true });
    data.append(FP16x16 { mag: 3049, sign: false });
    data.append(FP16x16 { mag: 23146, sign: true });
    data.append(FP16x16 { mag: 24433, sign: true });
    data.append(FP16x16 { mag: 190661, sign: false });
    data.append(FP16x16 { mag: 4019, sign: true });
    data.append(FP16x16 { mag: 70772, sign: false });
    data.append(FP16x16 { mag: 127015, sign: false });
    data.append(FP16x16 { mag: 19087, sign: true });
    data.append(FP16x16 { mag: 23993, sign: true });
    data.append(FP16x16 { mag: 11606, sign: true });
    data.append(FP16x16 { mag: 8249, sign: true });
    data.append(FP16x16 { mag: 69794, sign: false });
    data.append(FP16x16 { mag: 19075, sign: false });
    data.append(FP16x16 { mag: 53562, sign: false });
    data.append(FP16x16 { mag: 4281, sign: true });
    data.append(FP16x16 { mag: 13606, sign: true });
    data.append(FP16x16 { mag: 14990, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
