use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(3);
    shape.append(3);
    shape.append(1);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 1327160, sign: false });
    data.append(FP8x23 { mag: 11425140, sign: false });
    data.append(FP8x23 { mag: 9174538, sign: true });
    data.append(FP8x23 { mag: 1838060, sign: false });
    data.append(FP8x23 { mag: 10631878, sign: true });
    data.append(FP8x23 { mag: 11283422, sign: true });
    data.append(FP8x23 { mag: 652363, sign: false });
    data.append(FP8x23 { mag: 2113923, sign: false });
    data.append(FP8x23 { mag: 3033613, sign: true });
    data.append(FP8x23 { mag: 14034808, sign: false });
    data.append(FP8x23 { mag: 4803025, sign: false });
    data.append(FP8x23 { mag: 2062016, sign: true });
    data.append(FP8x23 { mag: 9606798, sign: true });
    data.append(FP8x23 { mag: 4232337, sign: true });
    data.append(FP8x23 { mag: 681468, sign: true });
    data.append(FP8x23 { mag: 13944945, sign: true });
    data.append(FP8x23 { mag: 7066023, sign: true });
    data.append(FP8x23 { mag: 7444007, sign: false });
    data.append(FP8x23 { mag: 1464875, sign: true });
    data.append(FP8x23 { mag: 5711684, sign: true });
    data.append(FP8x23 { mag: 7021604, sign: true });
    data.append(FP8x23 { mag: 13612406, sign: true });
    data.append(FP8x23 { mag: 15106801, sign: true });
    data.append(FP8x23 { mag: 2987753, sign: false });
    data.append(FP8x23 { mag: 1970786, sign: true });
    data.append(FP8x23 { mag: 7881602, sign: true });
    data.append(FP8x23 { mag: 364083, sign: true });
    data.append(FP8x23 { mag: 6157580, sign: true });
    data.append(FP8x23 { mag: 7242722, sign: false });
    data.append(FP8x23 { mag: 9527289, sign: true });
    data.append(FP8x23 { mag: 5205132, sign: false });
    data.append(FP8x23 { mag: 1177590, sign: false });
    data.append(FP8x23 { mag: 4317500, sign: true });
    data.append(FP8x23 { mag: 4550206, sign: false });
    data.append(FP8x23 { mag: 10877309, sign: false });
    data.append(FP8x23 { mag: 13442559, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
