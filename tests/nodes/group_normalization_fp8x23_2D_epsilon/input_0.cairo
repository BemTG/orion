use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 5335956, sign: true });
    data.append(FP8x23 { mag: 9003386, sign: false });
    data.append(FP8x23 { mag: 2208106, sign: false });
    data.append(FP8x23 { mag: 5040107, sign: true });
    data.append(FP8x23 { mag: 492011, sign: false });
    data.append(FP8x23 { mag: 986692, sign: false });
    data.append(FP8x23 { mag: 20889694, sign: true });
    data.append(FP8x23 { mag: 11512455, sign: true });
    data.append(FP8x23 { mag: 6624226, sign: false });
    data.append(FP8x23 { mag: 4386057, sign: false });
    data.append(FP8x23 { mag: 1136673, sign: false });
    data.append(FP8x23 { mag: 1230895, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
