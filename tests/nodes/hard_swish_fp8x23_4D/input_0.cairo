use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 11486910, sign: true });
    data.append(FP8x23 { mag: 24636618, sign: false });
    data.append(FP8x23 { mag: 16131755, sign: false });
    data.append(FP8x23 { mag: 11580350, sign: false });
    data.append(FP8x23 { mag: 2787207, sign: true });
    data.append(FP8x23 { mag: 10957847, sign: false });
    data.append(FP8x23 { mag: 16947560, sign: true });
    data.append(FP8x23 { mag: 184607, sign: false });
    data.append(FP8x23 { mag: 1649909, sign: false });
    data.append(FP8x23 { mag: 10388659, sign: true });
    data.append(FP8x23 { mag: 4095030, sign: true });
    data.append(FP8x23 { mag: 23990144, sign: true });
    data.append(FP8x23 { mag: 13501057, sign: false });
    data.append(FP8x23 { mag: 20800950, sign: false });
    data.append(FP8x23 { mag: 2105934, sign: false });
    data.append(FP8x23 { mag: 24494110, sign: true });
    data.append(FP8x23 { mag: 4224184, sign: false });
    data.append(FP8x23 { mag: 24130686, sign: true });
    data.append(FP8x23 { mag: 10639528, sign: true });
    data.append(FP8x23 { mag: 4204630, sign: false });
    data.append(FP8x23 { mag: 21032502, sign: false });
    data.append(FP8x23 { mag: 21280450, sign: false });
    data.append(FP8x23 { mag: 14567652, sign: true });
    data.append(FP8x23 { mag: 23445420, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
