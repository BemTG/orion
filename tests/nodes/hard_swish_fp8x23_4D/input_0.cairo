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
    data.append(FP8x23 { mag: 11861024, sign: false });
    data.append(FP8x23 { mag: 15987936, sign: false });
    data.append(FP8x23 { mag: 6245222, sign: true });
    data.append(FP8x23 { mag: 9652343, sign: false });
    data.append(FP8x23 { mag: 13681692, sign: false });
    data.append(FP8x23 { mag: 2696450, sign: true });
    data.append(FP8x23 { mag: 17687588, sign: false });
    data.append(FP8x23 { mag: 22952276, sign: true });
    data.append(FP8x23 { mag: 15427254, sign: true });
    data.append(FP8x23 { mag: 17534072, sign: true });
    data.append(FP8x23 { mag: 23675136, sign: true });
    data.append(FP8x23 { mag: 12726347, sign: false });
    data.append(FP8x23 { mag: 22772608, sign: false });
    data.append(FP8x23 { mag: 6628200, sign: false });
    data.append(FP8x23 { mag: 17790732, sign: true });
    data.append(FP8x23 { mag: 12013268, sign: true });
    data.append(FP8x23 { mag: 24896378, sign: false });
    data.append(FP8x23 { mag: 18256738, sign: true });
    data.append(FP8x23 { mag: 20687070, sign: false });
    data.append(FP8x23 { mag: 9789767, sign: true });
    data.append(FP8x23 { mag: 23638774, sign: true });
    data.append(FP8x23 { mag: 2499757, sign: true });
    data.append(FP8x23 { mag: 1172096, sign: false });
    data.append(FP8x23 { mag: 17830666, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
