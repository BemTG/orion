use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 3068710, sign: true });
    data.append(FP8x23 { mag: 9540600, sign: true });
    data.append(FP8x23 { mag: 2608521, sign: false });
    data.append(FP8x23 { mag: 9808462, sign: false });
    data.append(FP8x23 { mag: 7888071, sign: true });
    data.append(FP8x23 { mag: 9216630, sign: true });
    data.append(FP8x23 { mag: 664608, sign: true });
    data.append(FP8x23 { mag: 5730875, sign: true });
    data.append(FP8x23 { mag: 18476888, sign: true });
    data.append(FP8x23 { mag: 6831641, sign: false });
    data.append(FP8x23 { mag: 2183154, sign: false });
    data.append(FP8x23 { mag: 7659358, sign: false });
    data.append(FP8x23 { mag: 1874022, sign: false });
    data.append(FP8x23 { mag: 1457123, sign: false });
    data.append(FP8x23 { mag: 8416547, sign: true });
    data.append(FP8x23 { mag: 1233881, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
