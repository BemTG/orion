use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Array<Tensor<FP16x16>> {
    let mut sequence = ArrayTrait::new();

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(1);
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 39418, sign: false });
    data.append(FP16x16 { mag: 27877, sign: false });
    data.append(FP16x16 { mag: 41070, sign: false });
    data.append(FP16x16 { mag: 39605, sign: false });
    data.append(FP16x16 { mag: 51910, sign: false });
    data.append(FP16x16 { mag: 55540, sign: false });
    data.append(FP16x16 { mag: 49590, sign: false });
    data.append(FP16x16 { mag: 56123, sign: false });
    data.append(FP16x16 { mag: 59704, sign: false });
    data.append(FP16x16 { mag: 64520, sign: false });
    data.append(FP16x16 { mag: 62041, sign: false });
    data.append(FP16x16 { mag: 59582, sign: false });
    data.append(FP16x16 { mag: 62204, sign: false });
    data.append(FP16x16 { mag: 64393, sign: false });
    data.append(FP16x16 { mag: 65467, sign: false });
    data.append(FP16x16 { mag: 64797, sign: false });
    data.append(FP16x16 { mag: 64619, sign: false });
    data.append(FP16x16 { mag: 64996, sign: false });
    data.append(FP16x16 { mag: 65380, sign: false });
    data.append(FP16x16 { mag: 65533, sign: false });
    data.append(FP16x16 { mag: 65324, sign: false });
    data.append(FP16x16 { mag: 65316, sign: false });
    data.append(FP16x16 { mag: 65398, sign: false });
    data.append(FP16x16 { mag: 65511, sign: false });
    data.append(FP16x16 { mag: 65535, sign: false });
    data.append(FP16x16 { mag: 65467, sign: false });
    data.append(FP16x16 { mag: 65468, sign: false });
    data.append(FP16x16 { mag: 65493, sign: false });
    data.append(FP16x16 { mag: 65531, sign: false });
    data.append(FP16x16 { mag: 65535, sign: false });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 64797, sign: false });
    data.append(FP16x16 { mag: 64619, sign: false });
    data.append(FP16x16 { mag: 64996, sign: false });
    data.append(FP16x16 { mag: 65380, sign: false });
    data.append(FP16x16 { mag: 65533, sign: false });
    data.append(FP16x16 { mag: 65324, sign: false });
    data.append(FP16x16 { mag: 65316, sign: false });
    data.append(FP16x16 { mag: 65398, sign: false });
    data.append(FP16x16 { mag: 65511, sign: false });
    data.append(FP16x16 { mag: 65535, sign: false });
    data.append(FP16x16 { mag: 65467, sign: false });
    data.append(FP16x16 { mag: 65468, sign: false });
    data.append(FP16x16 { mag: 65493, sign: false });
    data.append(FP16x16 { mag: 65531, sign: false });
    data.append(FP16x16 { mag: 65535, sign: false });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    sequence
}
