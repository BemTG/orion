use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 3121862, sign: true });
    data.append(FP8x23 { mag: 24377580, sign: false });
    data.append(FP8x23 { mag: 13236253, sign: false });
    data.append(FP8x23 { mag: 8454592, sign: false });
    data.append(FP8x23 { mag: 1239257, sign: true });
    data.append(FP8x23 { mag: 7864588, sign: false });
    data.append(FP8x23 { mag: 2767235, sign: true });
    data.append(FP8x23 { mag: 92980, sign: false });
    data.append(FP8x23 { mag: 879040, sign: false });
    data.append(FP8x23 { mag: 3050067, sign: true });
    data.append(FP8x23 { mag: 1714339, sign: true });
    data.append(FP8x23 { mag: 560377, sign: true });
    data.append(FP8x23 { mag: 10372077, sign: false });
    data.append(FP8x23 { mag: 18997044, sign: false });
    data.append(FP8x23 { mag: 1141081, sign: false });
    data.append(FP8x23 { mag: 326892, sign: true });
    data.append(FP8x23 { mag: 2466615, sign: false });
    data.append(FP8x23 { mag: 496280, sign: true });
    data.append(FP8x23 { mag: 3070690, sign: true });
    data.append(FP8x23 { mag: 2453563, sign: false });
    data.append(FP8x23 { mag: 19305276, sign: false });
    data.append(FP8x23 { mag: 19637698, sign: false });
    data.append(FP8x23 { mag: 3067463, sign: true });
    data.append(FP8x23 { mag: 801396, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
