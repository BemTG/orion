use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 13046814, sign: true });
    data.append(FP8x23 { mag: 17619524, sign: true });
    data.append(FP8x23 { mag: 10833202, sign: true });
    data.append(FP8x23 { mag: 14869264, sign: true });
    data.append(FP8x23 { mag: 7983649, sign: true });
    data.append(FP8x23 { mag: 90546, sign: false });
    data.append(FP8x23 { mag: 6163406, sign: true });
    data.append(FP8x23 { mag: 1562675, sign: true });
    data.append(FP8x23 { mag: 2040971, sign: true });
    data.append(FP8x23 { mag: 11360608, sign: true });
    data.append(FP8x23 { mag: 4383010, sign: true });
    data.append(FP8x23 { mag: 1594327, sign: false });
    data.append(FP8x23 { mag: 9670559, sign: true });
    data.append(FP8x23 { mag: 18735284, sign: true });
    data.append(FP8x23 { mag: 12387203, sign: true });
    data.append(FP8x23 { mag: 11504010, sign: true });
    data.append(FP8x23 { mag: 6094216, sign: true });
    data.append(FP8x23 { mag: 6242606, sign: true });
    data.append(FP8x23 { mag: 5047260, sign: true });
    data.append(FP8x23 { mag: 4582027, sign: true });
    data.append(FP8x23 { mag: 9436015, sign: false });
    data.append(FP8x23 { mag: 22564112, sign: true });
    data.append(FP8x23 { mag: 17568256, sign: true });
    data.append(FP8x23 { mag: 3337188, sign: false });
    data.append(FP8x23 { mag: 18089220, sign: true });
    data.append(FP8x23 { mag: 10259306, sign: true });
    data.append(FP8x23 { mag: 13789114, sign: true });
    data.append(FP8x23 { mag: 13698670, sign: true });
    data.append(FP8x23 { mag: 5161665, sign: true });
    data.append(FP8x23 { mag: 6815934, sign: true });
    data.append(FP8x23 { mag: 7079479, sign: true });
    data.append(FP8x23 { mag: 822021, sign: true });
    data.append(FP8x23 { mag: 5020482, sign: false });
    data.append(FP8x23 { mag: 21880888, sign: true });
    data.append(FP8x23 { mag: 4757494, sign: true });
    data.append(FP8x23 { mag: 11159891, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
