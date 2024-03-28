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
    data.append(FP8x23 { mag: 697287, sign: false });
    data.append(FP8x23 { mag: 305853, sign: true });
    data.append(FP8x23 { mag: 526333, sign: true });
    data.append(FP8x23 { mag: 6524453, sign: false });
    data.append(FP8x23 { mag: 7533265, sign: true });
    data.append(FP8x23 { mag: 2781745, sign: true });
    data.append(FP8x23 { mag: 10737842, sign: true });
    data.append(FP8x23 { mag: 6547310, sign: true });
    data.append(FP8x23 { mag: 17634294, sign: true });
    data.append(FP8x23 { mag: 67657184, sign: false });
    data.append(FP8x23 { mag: 7977930, sign: false });
    data.append(FP8x23 { mag: 2904598, sign: true });
    data.append(FP8x23 { mag: 15034645, sign: true });
    data.append(FP8x23 { mag: 4707792, sign: true });
    data.append(FP8x23 { mag: 9115830, sign: true });
    data.append(FP8x23 { mag: 31528242, sign: true });
    data.append(FP8x23 { mag: 3199311, sign: true });
    data.append(FP8x23 { mag: 4162271, sign: false });
    data.append(FP8x23 { mag: 5850848, sign: true });
    data.append(FP8x23 { mag: 4395574, sign: false });
    data.append(FP8x23 { mag: 30984454, sign: false });
    data.append(FP8x23 { mag: 4873625, sign: true });
    data.append(FP8x23 { mag: 21381970, sign: false });
    data.append(FP8x23 { mag: 3337505, sign: true });
    data.append(FP8x23 { mag: 20970728, sign: true });
    data.append(FP8x23 { mag: 5096527, sign: true });
    data.append(FP8x23 { mag: 27771838, sign: false });
    data.append(FP8x23 { mag: 157967, sign: false });
    data.append(FP8x23 { mag: 7713674, sign: true });
    data.append(FP8x23 { mag: 13892416, sign: true });
    data.append(FP8x23 { mag: 10590048, sign: true });
    data.append(FP8x23 { mag: 3745579, sign: true });
    data.append(FP8x23 { mag: 4548798, sign: true });
    data.append(FP8x23 { mag: 1164113, sign: false });
    data.append(FP8x23 { mag: 8020439, sign: false });
    data.append(FP8x23 { mag: 11952859, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
