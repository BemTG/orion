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
    data.append(FP8x23 { mag: 45472404, sign: true });
    data.append(FP8x23 { mag: 43212160, sign: true });
    data.append(FP8x23 { mag: 46516216, sign: true });
    data.append(FP8x23 { mag: 33373830, sign: true });
    data.append(FP8x23 { mag: 28189642, sign: true });
    data.append(FP8x23 { mag: 47024884, sign: true });
    data.append(FP8x23 { mag: 43653224, sign: true });
    data.append(FP8x23 { mag: 29243500, sign: true });
    data.append(FP8x23 { mag: 39683136, sign: true });
    data.append(FP8x23 { mag: 27449540, sign: true });
    data.append(FP8x23 { mag: 25276808, sign: true });
    data.append(FP8x23 { mag: 47278224, sign: true });
    data.append(FP8x23 { mag: 46835940, sign: true });
    data.append(FP8x23 { mag: 33809744, sign: true });
    data.append(FP8x23 { mag: 48953700, sign: true });
    data.append(FP8x23 { mag: 45328012, sign: true });
    data.append(FP8x23 { mag: 32235858, sign: true });
    data.append(FP8x23 { mag: 44113384, sign: true });
    data.append(FP8x23 { mag: 46270808, sign: true });
    data.append(FP8x23 { mag: 28652994, sign: true });
    data.append(FP8x23 { mag: 45169384, sign: true });
    data.append(FP8x23 { mag: 26883722, sign: true });
    data.append(FP8x23 { mag: 49120116, sign: true });
    data.append(FP8x23 { mag: 45273176, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
