use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 24106790, sign: true });
    data.append(FP8x23 { mag: 5849987, sign: true });
    data.append(FP8x23 { mag: 14670506, sign: true });
    data.append(FP8x23 { mag: 1700485, sign: true });
    data.append(FP8x23 { mag: 10968212, sign: true });
    data.append(FP8x23 { mag: 5869172, sign: false });
    data.append(FP8x23 { mag: 11590380, sign: false });
    data.append(FP8x23 { mag: 7205000, sign: false });
    data.append(FP8x23 { mag: 12602334, sign: true });
    data.append(FP8x23 { mag: 20377758, sign: true });
    data.append(FP8x23 { mag: 7134174, sign: false });
    data.append(FP8x23 { mag: 18490190, sign: true });
    data.append(FP8x23 { mag: 240976, sign: false });
    data.append(FP8x23 { mag: 10574422, sign: false });
    data.append(FP8x23 { mag: 9600946, sign: false });
    data.append(FP8x23 { mag: 756579, sign: false });
    data.append(FP8x23 { mag: 24996714, sign: true });
    data.append(FP8x23 { mag: 9405727, sign: true });
    data.append(FP8x23 { mag: 603728, sign: false });
    data.append(FP8x23 { mag: 9600399, sign: true });
    data.append(FP8x23 { mag: 523355, sign: true });
    data.append(FP8x23 { mag: 12115159, sign: true });
    data.append(FP8x23 { mag: 10930180, sign: false });
    data.append(FP8x23 { mag: 11862640, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
