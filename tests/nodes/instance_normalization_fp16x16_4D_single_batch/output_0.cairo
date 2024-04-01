use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 44989, sign: true });
    data.append(FP16x16 { mag: 6663, sign: true });
    data.append(FP16x16 { mag: 62379, sign: true });
    data.append(FP16x16 { mag: 38227, sign: true });
    data.append(FP16x16 { mag: 30228, sign: true });
    data.append(FP16x16 { mag: 86759, sign: false });
    data.append(FP16x16 { mag: 63932, sign: false });
    data.append(FP16x16 { mag: 2506, sign: false });
    data.append(FP16x16 { mag: 6134, sign: false });
    data.append(FP16x16 { mag: 33877, sign: false });
    data.append(FP16x16 { mag: 20182, sign: false });
    data.append(FP16x16 { mag: 62021, sign: false });
    data.append(FP16x16 { mag: 38206, sign: true });
    data.append(FP16x16 { mag: 46250, sign: true });
    data.append(FP16x16 { mag: 18744, sign: false });
    data.append(FP16x16 { mag: 9476, sign: true });
    data.append(FP16x16 { mag: 38989, sign: true });
    data.append(FP16x16 { mag: 600, sign: true });
    data.append(FP16x16 { mag: 20726, sign: false });
    data.append(FP16x16 { mag: 21181, sign: true });
    data.append(FP16x16 { mag: 21905, sign: true });
    data.append(FP16x16 { mag: 129739, sign: true });
    data.append(FP16x16 { mag: 24026, sign: true });
    data.append(FP16x16 { mag: 94531, sign: false });
    data.append(FP16x16 { mag: 40376, sign: true });
    data.append(FP16x16 { mag: 59967, sign: false });
    data.append(FP16x16 { mag: 85005, sign: false });
    data.append(FP16x16 { mag: 57318, sign: true });
    data.append(FP16x16 { mag: 109636, sign: true });
    data.append(FP16x16 { mag: 41669, sign: false });
    data.append(FP16x16 { mag: 36983, sign: true });
    data.append(FP16x16 { mag: 44537, sign: true });
    data.append(FP16x16 { mag: 13127, sign: false });
    data.append(FP16x16 { mag: 33593, sign: false });
    data.append(FP16x16 { mag: 114194, sign: true });
    data.append(FP16x16 { mag: 59745, sign: false });
    data.append(FP16x16 { mag: 28944, sign: false });
    data.append(FP16x16 { mag: 35611, sign: true });
    data.append(FP16x16 { mag: 189756, sign: true });
    data.append(FP16x16 { mag: 140165, sign: false });
    data.append(FP16x16 { mag: 286714, sign: true });
    data.append(FP16x16 { mag: 121198, sign: true });
    data.append(FP16x16 { mag: 193099, sign: true });
    data.append(FP16x16 { mag: 72347, sign: false });
    data.append(FP16x16 { mag: 348700, sign: true });
    data.append(FP16x16 { mag: 57138, sign: true });
    data.append(FP16x16 { mag: 69029, sign: true });
    data.append(FP16x16 { mag: 121556, sign: true });
    data.append(FP16x16 { mag: 48322, sign: false });
    data.append(FP16x16 { mag: 20575, sign: true });
    data.append(FP16x16 { mag: 257835, sign: true });
    data.append(FP16x16 { mag: 137880, sign: false });
    data.append(FP16x16 { mag: 126404, sign: false });
    data.append(FP16x16 { mag: 32959, sign: false });
    data.append(FP16x16 { mag: 26475, sign: false });
    data.append(FP16x16 { mag: 21732, sign: true });
    data.append(FP16x16 { mag: 69586, sign: true });
    data.append(FP16x16 { mag: 279224, sign: true });
    data.append(FP16x16 { mag: 209879, sign: false });
    data.append(FP16x16 { mag: 1714, sign: true });
    TensorTrait::new(shape.span(), data.span())
}