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
    data.append(FP8x23 { mag: 12477934, sign: true });
    data.append(FP8x23 { mag: 23323292, sign: true });
    data.append(FP8x23 { mag: 2316698, sign: true });
    data.append(FP8x23 { mag: 8061404, sign: false });
    data.append(FP8x23 { mag: 18558290, sign: false });
    data.append(FP8x23 { mag: 21856868, sign: false });
    data.append(FP8x23 { mag: 1264612, sign: false });
    data.append(FP8x23 { mag: 2265726, sign: true });
    data.append(FP8x23 { mag: 22161508, sign: true });
    data.append(FP8x23 { mag: 8573826, sign: true });
    data.append(FP8x23 { mag: 8429835, sign: false });
    data.append(FP8x23 { mag: 5471394, sign: true });
    data.append(FP8x23 { mag: 19723644, sign: false });
    data.append(FP8x23 { mag: 21496874, sign: false });
    data.append(FP8x23 { mag: 1698017, sign: false });
    data.append(FP8x23 { mag: 4124693, sign: true });
    data.append(FP8x23 { mag: 19547260, sign: true });
    data.append(FP8x23 { mag: 21320010, sign: true });
    data.append(FP8x23 { mag: 3093450, sign: false });
    data.append(FP8x23 { mag: 5437607, sign: false });
    data.append(FP8x23 { mag: 19362294, sign: false });
    data.append(FP8x23 { mag: 19087774, sign: false });
    data.append(FP8x23 { mag: 3419897, sign: false });
    data.append(FP8x23 { mag: 942617, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
