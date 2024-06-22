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
    data.append(FP16x16 { mag: 65535, sign: true });
    data.append(FP16x16 { mag: 65506, sign: true });
    data.append(FP16x16 { mag: 65204, sign: true });
    data.append(FP16x16 { mag: 20450, sign: true });
    data.append(FP16x16 { mag: 21079, sign: true });
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 29532, sign: true });
    data.append(FP16x16 { mag: 50798, sign: false });
    data.append(FP16x16 { mag: 64455, sign: true });
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 63255, sign: false });
    data.append(FP16x16 { mag: 64442, sign: false });
    data.append(FP16x16 { mag: 65518, sign: true });
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 65534, sign: false });
    data.append(FP16x16 { mag: 52555, sign: false });
    data.append(FP16x16 { mag: 65535, sign: true });
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 64333, sign: false });
    data.append(FP16x16 { mag: 65535, sign: true });
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 65467, sign: false });
    data.append(FP16x16 { mag: 65536, sign: true });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 65534, sign: false });
    data.append(FP16x16 { mag: 52555, sign: false });
    data.append(FP16x16 { mag: 65535, sign: true });
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 64333, sign: false });
    data.append(FP16x16 { mag: 65535, sign: true });
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 65536, sign: true });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 65467, sign: false });
    data.append(FP16x16 { mag: 65536, sign: true });

    sequence.append(TensorTrait::new(shape.span(), data.span()));

    sequence
}
