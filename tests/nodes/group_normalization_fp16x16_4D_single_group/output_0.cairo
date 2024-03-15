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
    data.append(FP16x16 { mag: 89959, sign: true });
    data.append(FP16x16 { mag: 75764, sign: true });
    data.append(FP16x16 { mag: 102225, sign: true });
    data.append(FP16x16 { mag: 94674, sign: true });
    data.append(FP16x16 { mag: 56129, sign: false });
    data.append(FP16x16 { mag: 96681, sign: false });
    data.append(FP16x16 { mag: 74628, sign: false });
    data.append(FP16x16 { mag: 97963, sign: false });
    data.append(FP16x16 { mag: 68928, sign: true });
    data.append(FP16x16 { mag: 126823, sign: true });
    data.append(FP16x16 { mag: 67352, sign: true });
    data.append(FP16x16 { mag: 135145, sign: true });
    data.append(FP16x16 { mag: 84678, sign: false });
    data.append(FP16x16 { mag: 97466, sign: false });
    data.append(FP16x16 { mag: 71734, sign: false });
    data.append(FP16x16 { mag: 98596, sign: false });
    data.append(FP16x16 { mag: 157425, sign: true });
    data.append(FP16x16 { mag: 123868, sign: true });
    data.append(FP16x16 { mag: 145444, sign: true });
    data.append(FP16x16 { mag: 75090, sign: true });
    data.append(FP16x16 { mag: 118747, sign: false });
    data.append(FP16x16 { mag: 114321, sign: false });
    data.append(FP16x16 { mag: 99739, sign: false });
    data.append(FP16x16 { mag: 98378, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
