use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(5);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 26638, sign: true });
    data.append(FP16x16 { mag: 1631, sign: true });
    data.append(FP16x16 { mag: 10738, sign: true });
    data.append(FP16x16 { mag: 13081, sign: true });
    data.append(FP16x16 { mag: 24622, sign: false });
    data.append(FP16x16 { mag: 20261, sign: false });
    data.append(FP16x16 { mag: 29131, sign: false });
    data.append(FP16x16 { mag: 8638, sign: true });
    data.append(FP16x16 { mag: 27453, sign: false });
    data.append(FP16x16 { mag: 48569, sign: false });
    data.append(FP16x16 { mag: 23218, sign: true });
    data.append(FP16x16 { mag: 80239, sign: false });
    data.append(FP16x16 { mag: 11043, sign: true });
    data.append(FP16x16 { mag: 20427, sign: true });
    data.append(FP16x16 { mag: 59582, sign: true });
    data.append(FP16x16 { mag: 39660, sign: false });
    data.append(FP16x16 { mag: 48915, sign: true });
    data.append(FP16x16 { mag: 13174, sign: false });
    data.append(FP16x16 { mag: 16280, sign: true });
    data.append(FP16x16 { mag: 95062, sign: false });
    data.append(FP16x16 { mag: 27098, sign: true });
    data.append(FP16x16 { mag: 10767, sign: true });
    data.append(FP16x16 { mag: 121528, sign: false });
    data.append(FP16x16 { mag: 6457, sign: false });
    data.append(FP16x16 { mag: 57011, sign: false });
    data.append(FP16x16 { mag: 27892, sign: true });
    data.append(FP16x16 { mag: 60302, sign: false });
    data.append(FP16x16 { mag: 85321, sign: false });
    data.append(FP16x16 { mag: 87077, sign: true });
    data.append(FP16x16 { mag: 10628, sign: true });
    data.append(FP16x16 { mag: 35127, sign: false });
    data.append(FP16x16 { mag: 72841, sign: false });
    data.append(FP16x16 { mag: 25061, sign: true });
    data.append(FP16x16 { mag: 33618, sign: true });
    data.append(FP16x16 { mag: 72952, sign: true });
    data.append(FP16x16 { mag: 38948, sign: false });
    data.append(FP16x16 { mag: 48153, sign: true });
    data.append(FP16x16 { mag: 89003, sign: true });
    data.append(FP16x16 { mag: 7721, sign: false });
    data.append(FP16x16 { mag: 38866, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
