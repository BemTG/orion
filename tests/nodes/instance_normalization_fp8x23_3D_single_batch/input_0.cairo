use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 3984439, sign: false });
    data.append(FP8x23 { mag: 16641106, sign: false });
    data.append(FP8x23 { mag: 8518488, sign: false });
    data.append(FP8x23 { mag: 511783, sign: true });
    data.append(FP8x23 { mag: 12986276, sign: false });
    data.append(FP8x23 { mag: 11766351, sign: true });
    data.append(FP8x23 { mag: 3648139, sign: false });
    data.append(FP8x23 { mag: 1293554, sign: false });
    data.append(FP8x23 { mag: 493204, sign: true });
    data.append(FP8x23 { mag: 2202479, sign: true });
    data.append(FP8x23 { mag: 2670284, sign: false });
    data.append(FP8x23 { mag: 1566597, sign: true });
    data.append(FP8x23 { mag: 4497214, sign: true });
    data.append(FP8x23 { mag: 20159974, sign: true });
    data.append(FP8x23 { mag: 3481733, sign: true });
    data.append(FP8x23 { mag: 10801086, sign: false });
    data.append(FP8x23 { mag: 260799, sign: true });
    data.append(FP8x23 { mag: 2007020, sign: true });
    data.append(FP8x23 { mag: 12266421, sign: true });
    data.append(FP8x23 { mag: 1947209, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
