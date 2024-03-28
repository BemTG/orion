use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 7079866, sign: false });
    data.append(FP8x23 { mag: 16703195, sign: false });
    data.append(FP8x23 { mag: 15614848, sign: false });
    data.append(FP8x23 { mag: 13942373, sign: false });
    data.append(FP8x23 { mag: 812987, sign: false });
    data.append(FP8x23 { mag: 3246030, sign: true });
    data.append(FP8x23 { mag: 555442, sign: false });
    data.append(FP8x23 { mag: 1601133, sign: false });
    data.append(FP8x23 { mag: 18946676, sign: false });
    data.append(FP8x23 { mag: 8411130, sign: false });
    data.append(FP8x23 { mag: 13640624, sign: false });
    data.append(FP8x23 { mag: 20194632, sign: false });
    data.append(FP8x23 { mag: 1406277, sign: false });
    data.append(FP8x23 { mag: 7399901, sign: true });
    data.append(FP8x23 { mag: 1283121, sign: false });
    data.append(FP8x23 { mag: 1387846, sign: false });
    data.append(FP8x23 { mag: 13119846, sign: false });
    data.append(FP8x23 { mag: 16988698, sign: false });
    data.append(FP8x23 { mag: 24203872, sign: false });
    data.append(FP8x23 { mag: 12923297, sign: false });
    data.append(FP8x23 { mag: 1795206, sign: false });
    data.append(FP8x23 { mag: 1429168, sign: true });
    data.append(FP8x23 { mag: 1048008, sign: false });
    data.append(FP8x23 { mag: 704187, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
