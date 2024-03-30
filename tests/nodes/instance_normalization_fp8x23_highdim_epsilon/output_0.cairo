use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(3);
    shape.append(3);
    shape.append(1);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 6264764, sign: false });
    data.append(FP8x23 { mag: 16216584, sign: false });
    data.append(FP8x23 { mag: 4084929, sign: true });
    data.append(FP8x23 { mag: 4749186, sign: true });
    data.append(FP8x23 { mag: 1269172, sign: false });
    data.append(FP8x23 { mag: 1583626, sign: false });
    data.append(FP8x23 { mag: 3633650, sign: true });
    data.append(FP8x23 { mag: 14020066, sign: true });
    data.append(FP8x23 { mag: 22560344, sign: false });
    data.append(FP8x23 { mag: 16454498, sign: false });
    data.append(FP8x23 { mag: 5167604, sign: false });
    data.append(FP8x23 { mag: 3225682, sign: true });
    data.append(FP8x23 { mag: 2718805, sign: false });
    data.append(FP8x23 { mag: 1059459, sign: true });
    data.append(FP8x23 { mag: 3555733, sign: true });
    data.append(FP8x23 { mag: 26288776, sign: false });
    data.append(FP8x23 { mag: 8290853, sign: false });
    data.append(FP8x23 { mag: 29673008, sign: true });
    data.append(FP8x23 { mag: 14113574, sign: false });
    data.append(FP8x23 { mag: 3741107, sign: false });
    data.append(FP8x23 { mag: 541738, sign: false });
    data.append(FP8x23 { mag: 1227375, sign: false });
    data.append(FP8x23 { mag: 1779250, sign: false });
    data.append(FP8x23 { mag: 4903010, sign: true });
    data.append(FP8x23 { mag: 6714546, sign: true });
    data.append(FP8x23 { mag: 27686844, sign: false });
    data.append(FP8x23 { mag: 16065675, sign: true });
    data.append(FP8x23 { mag: 2364869, sign: false });
    data.append(FP8x23 { mag: 17463446, sign: false });
    data.append(FP8x23 { mag: 1431896, sign: true });
    data.append(FP8x23 { mag: 3678501, sign: true });
    data.append(FP8x23 { mag: 962065, sign: true });
    data.append(FP8x23 { mag: 2744181, sign: false });
    data.append(FP8x23 { mag: 7260082, sign: true });
    data.append(FP8x23 { mag: 21734232, sign: true });
    data.append(FP8x23 { mag: 33900940, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
