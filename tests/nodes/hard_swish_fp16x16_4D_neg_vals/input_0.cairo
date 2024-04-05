use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 221563, sign: true });
    data.append(FP16x16 { mag: 285818, sign: true });
    data.append(FP16x16 { mag: 317348, sign: true });
    data.append(FP16x16 { mag: 364602, sign: true });
    data.append(FP16x16 { mag: 318706, sign: true });
    data.append(FP16x16 { mag: 368250, sign: true });
    data.append(FP16x16 { mag: 335549, sign: true });
    data.append(FP16x16 { mag: 205137, sign: true });
    data.append(FP16x16 { mag: 313250, sign: true });
    data.append(FP16x16 { mag: 216023, sign: true });
    data.append(FP16x16 { mag: 278332, sign: true });
    data.append(FP16x16 { mag: 369336, sign: true });
    data.append(FP16x16 { mag: 373196, sign: true });
    data.append(FP16x16 { mag: 246183, sign: true });
    data.append(FP16x16 { mag: 206006, sign: true });
    data.append(FP16x16 { mag: 302135, sign: true });
    data.append(FP16x16 { mag: 389029, sign: true });
    data.append(FP16x16 { mag: 269231, sign: true });
    data.append(FP16x16 { mag: 339983, sign: true });
    data.append(FP16x16 { mag: 254936, sign: true });
    data.append(FP16x16 { mag: 389601, sign: true });
    data.append(FP16x16 { mag: 220984, sign: true });
    data.append(FP16x16 { mag: 293048, sign: true });
    data.append(FP16x16 { mag: 346968, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
