use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(5);
    shape.append(1);
    shape.append(2);
    shape.append(2);
    shape.append(1);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 26559, sign: true });
    data.append(FP16x16 { mag: 72936, sign: false });
    data.append(FP16x16 { mag: 41522, sign: false });
    data.append(FP16x16 { mag: 15083, sign: true });
    data.append(FP16x16 { mag: 5117, sign: true });
    data.append(FP16x16 { mag: 21476, sign: false });
    data.append(FP16x16 { mag: 7333, sign: true });
    data.append(FP16x16 { mag: 68653, sign: false });
    data.append(FP16x16 { mag: 35793, sign: true });
    data.append(FP16x16 { mag: 32148, sign: true });
    data.append(FP16x16 { mag: 47927, sign: true });
    data.append(FP16x16 { mag: 44798, sign: true });
    data.append(FP16x16 { mag: 84246, sign: false });
    data.append(FP16x16 { mag: 26595, sign: true });
    data.append(FP16x16 { mag: 41319, sign: false });
    data.append(FP16x16 { mag: 5357, sign: false });
    data.append(FP16x16 { mag: 72564, sign: false });
    data.append(FP16x16 { mag: 98318, sign: true });
    data.append(FP16x16 { mag: 185880, sign: false });
    data.append(FP16x16 { mag: 74257, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
