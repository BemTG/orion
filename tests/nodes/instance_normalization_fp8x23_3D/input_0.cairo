use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 14549376, sign: true });
    data.append(FP8x23 { mag: 702662, sign: false });
    data.append(FP8x23 { mag: 5614823, sign: false });
    data.append(FP8x23 { mag: 2623110, sign: true });
    data.append(FP8x23 { mag: 3591749, sign: false });
    data.append(FP8x23 { mag: 7945816, sign: true });
    data.append(FP8x23 { mag: 14590718, sign: true });
    data.append(FP8x23 { mag: 9169534, sign: true });
    data.append(FP8x23 { mag: 6090125, sign: true });
    data.append(FP8x23 { mag: 16111772, sign: false });
    data.append(FP8x23 { mag: 4009217, sign: false });
    data.append(FP8x23 { mag: 8713323, sign: true });
    data.append(FP8x23 { mag: 1095219, sign: false });
    data.append(FP8x23 { mag: 1990383, sign: true });
    data.append(FP8x23 { mag: 10096565, sign: false });
    data.append(FP8x23 { mag: 3587816, sign: false });
    data.append(FP8x23 { mag: 7715080, sign: true });
    data.append(FP8x23 { mag: 8773725, sign: false });
    data.append(FP8x23 { mag: 787227, sign: true });
    data.append(FP8x23 { mag: 1668789, sign: false });
    data.append(FP8x23 { mag: 6240906, sign: false });
    data.append(FP8x23 { mag: 5171422, sign: true });
    data.append(FP8x23 { mag: 2869263, sign: false });
    data.append(FP8x23 { mag: 2641286, sign: true });
    data.append(FP8x23 { mag: 1160592, sign: false });
    data.append(FP8x23 { mag: 6240712, sign: false });
    data.append(FP8x23 { mag: 585650, sign: false });
    data.append(FP8x23 { mag: 9213034, sign: true });
    data.append(FP8x23 { mag: 9751086, sign: false });
    data.append(FP8x23 { mag: 2925426, sign: true });
    data.append(FP8x23 { mag: 1871730, sign: false });
    data.append(FP8x23 { mag: 2406549, sign: true });
    data.append(FP8x23 { mag: 5485310, sign: false });
    data.append(FP8x23 { mag: 6067397, sign: false });
    data.append(FP8x23 { mag: 1285611, sign: false });
    data.append(FP8x23 { mag: 3269313, sign: true });
    data.append(FP8x23 { mag: 10264855, sign: false });
    data.append(FP8x23 { mag: 7420255, sign: false });
    data.append(FP8x23 { mag: 9605688, sign: true });
    data.append(FP8x23 { mag: 16940402, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
