use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_3() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(30);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 45875, sign: true });
    data.append(FP16x16 { mag: 45875, sign: true });
    data.append(FP16x16 { mag: 26214, sign: true });
    data.append(FP16x16 { mag: 19660, sign: false });
    data.append(FP16x16 { mag: 32768, sign: false });
    data.append(FP16x16 { mag: 45875, sign: true });
    data.append(FP16x16 { mag: 72089, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 6553, sign: false });
    data.append(FP16x16 { mag: 26214, sign: false });
    data.append(FP16x16 { mag: 13107, sign: false });
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 19660, sign: false });
    data.append(FP16x16 { mag: 39321, sign: true });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 19660, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 13107, sign: false });
    data.append(FP16x16 { mag: 13107, sign: true });
    data.append(FP16x16 { mag: 78643, sign: true });
    data.append(FP16x16 { mag: 19660, sign: true });
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 6553, sign: false });
    data.append(FP16x16 { mag: 6553, sign: true });
    data.append(FP16x16 { mag: 26214, sign: true });
    data.append(FP16x16 { mag: 6553, sign: true });
    data.append(FP16x16 { mag: 19660, sign: true });
    data.append(FP16x16 { mag: 32768, sign: false });
    data.append(FP16x16 { mag: 26214, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
