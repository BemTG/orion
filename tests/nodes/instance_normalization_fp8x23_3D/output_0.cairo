use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 7991319, sign: true });
    data.append(FP8x23 { mag: 5240537, sign: false });
    data.append(FP8x23 { mag: 137734, sign: false });
    data.append(FP8x23 { mag: 5371082, sign: true });
    data.append(FP8x23 { mag: 1677998, sign: false });
    data.append(FP8x23 { mag: 8473714, sign: true });
    data.append(FP8x23 { mag: 8021220, sign: true });
    data.append(FP8x23 { mag: 2707131, sign: true });
    data.append(FP8x23 { mag: 7933923, sign: true });
    data.append(FP8x23 { mag: 8488927, sign: true });
    data.append(FP8x23 { mag: 997025, sign: false });
    data.append(FP8x23 { mag: 9552019, sign: false });
    data.append(FP8x23 { mag: 10537557, sign: false });
    data.append(FP8x23 { mag: 2321570, sign: true });
    data.append(FP8x23 { mag: 19590576, sign: true });
    data.append(FP8x23 { mag: 4841690, sign: true });
    data.append(FP8x23 { mag: 3318827, sign: false });
    data.append(FP8x23 { mag: 2886071, sign: true });
    data.append(FP8x23 { mag: 723744, sign: true });
    data.append(FP8x23 { mag: 2379263, sign: true });
    data.append(FP8x23 { mag: 8519998, sign: true });
    data.append(FP8x23 { mag: 6447635, sign: false });
    data.append(FP8x23 { mag: 665824, sign: true });
    data.append(FP8x23 { mag: 2789346, sign: true });
    data.append(FP8x23 { mag: 778595, sign: true });
    data.append(FP8x23 { mag: 7903633, sign: true });
    data.append(FP8x23 { mag: 4412554, sign: true });
    data.append(FP8x23 { mag: 10962551, sign: true });
    data.append(FP8x23 { mag: 6448972, sign: true });
    data.append(FP8x23 { mag: 5897205, sign: true });
    data.append(FP8x23 { mag: 10727770, sign: false });
    data.append(FP8x23 { mag: 8639397, sign: true });
    data.append(FP8x23 { mag: 16750217, sign: true });
    data.append(FP8x23 { mag: 3283079, sign: false });
    data.append(FP8x23 { mag: 10553220, sign: false });
    data.append(FP8x23 { mag: 1620742, sign: true });
    data.append(FP8x23 { mag: 4193355, sign: true });
    data.append(FP8x23 { mag: 1445613, sign: true });
    data.append(FP8x23 { mag: 3775722, sign: true });
    data.append(FP8x23 { mag: 3523490, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
