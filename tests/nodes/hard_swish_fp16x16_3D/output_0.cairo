use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 17330, sign: true });
    data.append(FP16x16 { mag: 34998, sign: false });
    data.append(FP16x16 { mag: 191687, sign: false });
    data.append(FP16x16 { mag: 92033, sign: false });
    data.append(FP16x16 { mag: 21259, sign: true });
    data.append(FP16x16 { mag: 891, sign: true });
    data.append(FP16x16 { mag: 125659, sign: false });
    data.append(FP16x16 { mag: 23550, sign: true });
    data.append(FP16x16 { mag: 189211, sign: false });
    data.append(FP16x16 { mag: 16006, sign: false });
    data.append(FP16x16 { mag: 76591, sign: false });
    data.append(FP16x16 { mag: 62643, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
