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
    data.append(FP8x23 { mag: 8394466, sign: false });
    data.append(FP8x23 { mag: 8773015, sign: true });
    data.append(FP8x23 { mag: 2152457, sign: true });
    data.append(FP8x23 { mag: 4994877, sign: false });
    data.append(FP8x23 { mag: 4150851, sign: true });
    data.append(FP8x23 { mag: 8694712, sign: true });
    data.append(FP8x23 { mag: 7703355, sign: true });
    data.append(FP8x23 { mag: 3939128, sign: false });
    data.append(FP8x23 { mag: 7512099, sign: true });
    data.append(FP8x23 { mag: 8728041, sign: true });
    data.append(FP8x23 { mag: 4961683, sign: false });
    data.append(FP8x23 { mag: 10446258, sign: false });
    data.append(FP8x23 { mag: 11078083, sign: false });
    data.append(FP8x23 { mag: 2834143, sign: false });
    data.append(FP8x23 { mag: 8236953, sign: true });
    data.append(FP8x23 { mag: 661790, sign: true });
    data.append(FP8x23 { mag: 13247221, sign: true });
    data.append(FP8x23 { mag: 3677812, sign: true });
    data.append(FP8x23 { mag: 7012628, sign: true });
    data.append(FP8x23 { mag: 4459429, sign: true });
    data.append(FP8x23 { mag: 739168, sign: false });
    data.append(FP8x23 { mag: 5999072, sign: true });
    data.append(FP8x23 { mag: 2796682, sign: true });
    data.append(FP8x23 { mag: 1840699, sign: true });
    data.append(FP8x23 { mag: 2745914, sign: true });
    data.append(FP8x23 { mag: 7239132, sign: true });
    data.append(FP8x23 { mag: 5510818, sign: false });
    data.append(FP8x23 { mag: 18410764, sign: true });
    data.append(FP8x23 { mag: 1926488, sign: true });
    data.append(FP8x23 { mag: 88646, sign: false });
    data.append(FP8x23 { mag: 4625305, sign: false });
    data.append(FP8x23 { mag: 3280172, sign: true });
    data.append(FP8x23 { mag: 6590925, sign: true });
    data.append(FP8x23 { mag: 1586459, sign: false });
    data.append(FP8x23 { mag: 4554055, sign: false });
    data.append(FP8x23 { mag: 2515751, sign: true });
    data.append(FP8x23 { mag: 6726694, sign: false });
    data.append(FP8x23 { mag: 3144926, sign: true });
    data.append(FP8x23 { mag: 5226295, sign: false });
    data.append(FP8x23 { mag: 20997078, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
