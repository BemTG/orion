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
    data.append(FP16x16 { mag: 29659, sign: true });
    data.append(FP16x16 { mag: 47308, sign: true });
    data.append(FP16x16 { mag: 24739, sign: true });
    data.append(FP16x16 { mag: 11703, sign: false });
    data.append(FP16x16 { mag: 36033, sign: true });
    data.append(FP16x16 { mag: 7114, sign: true });
    data.append(FP16x16 { mag: 62598, sign: true });
    data.append(FP16x16 { mag: 4531, sign: true });
    data.append(FP16x16 { mag: 9746, sign: false });
    data.append(FP16x16 { mag: 8526, sign: true });
    data.append(FP16x16 { mag: 967, sign: true });
    data.append(FP16x16 { mag: 65051, sign: true });
    data.append(FP16x16 { mag: 590, sign: true });
    data.append(FP16x16 { mag: 6302, sign: false });
    data.append(FP16x16 { mag: 1178, sign: true });
    data.append(FP16x16 { mag: 29731, sign: true });
    data.append(FP16x16 { mag: 65420, sign: true });
    data.append(FP16x16 { mag: 24786, sign: true });
    data.append(FP16x16 { mag: 14910, sign: false });
    data.append(FP16x16 { mag: 36129, sign: true });
    data.append(FP16x16 { mag: 7127, sign: true });
    data.append(FP16x16 { mag: 65533, sign: true });
    data.append(FP16x16 { mag: 4540, sign: true });
    data.append(FP16x16 { mag: 12722, sign: false });
    data.append(FP16x16 { mag: 8540, sign: true });
    data.append(FP16x16 { mag: 969, sign: true });
    data.append(FP16x16 { mag: 65535, sign: true });
    data.append(FP16x16 { mag: 592, sign: true });
    data.append(FP16x16 { mag: 8312, sign: false });
    data.append(FP16x16 { mag: 1180, sign: true });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 29731, sign: true });
    data.append(FP16x16 { mag: 65420, sign: true });
    data.append(FP16x16 { mag: 24786, sign: true });
    data.append(FP16x16 { mag: 14910, sign: false });
    data.append(FP16x16 { mag: 36129, sign: true });
    data.append(FP16x16 { mag: 7127, sign: true });
    data.append(FP16x16 { mag: 65533, sign: true });
    data.append(FP16x16 { mag: 4540, sign: true });
    data.append(FP16x16 { mag: 12722, sign: false });
    data.append(FP16x16 { mag: 8540, sign: true });
    data.append(FP16x16 { mag: 969, sign: true });
    data.append(FP16x16 { mag: 65535, sign: true });
    data.append(FP16x16 { mag: 592, sign: true });
    data.append(FP16x16 { mag: 8312, sign: false });
    data.append(FP16x16 { mag: 1180, sign: true });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    sequence
}
