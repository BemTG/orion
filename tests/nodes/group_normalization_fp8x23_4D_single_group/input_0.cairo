use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 57729, sign: false });
    data.append(FP8x23 { mag: 16942650, sign: false });
    data.append(FP8x23 { mag: 8116126, sign: true });
    data.append(FP8x23 { mag: 6787202, sign: false });
    data.append(FP8x23 { mag: 9144519, sign: false });
    data.append(FP8x23 { mag: 22680598, sign: true });
    data.append(FP8x23 { mag: 1969880, sign: false });
    data.append(FP8x23 { mag: 16164286, sign: true });
    data.append(FP8x23 { mag: 1099290, sign: true });
    data.append(FP8x23 { mag: 9428410, sign: true });
    data.append(FP8x23 { mag: 3192410, sign: true });
    data.append(FP8x23 { mag: 2149638, sign: false });
    data.append(FP8x23 { mag: 10534561, sign: true });
    data.append(FP8x23 { mag: 18548608, sign: false });
    data.append(FP8x23 { mag: 1818510, sign: true });
    data.append(FP8x23 { mag: 4652137, sign: true });
    data.append(FP8x23 { mag: 1722254, sign: false });
    data.append(FP8x23 { mag: 2230457, sign: false });
    data.append(FP8x23 { mag: 1863336, sign: true });
    data.append(FP8x23 { mag: 3456656, sign: true });
    data.append(FP8x23 { mag: 8204760, sign: false });
    data.append(FP8x23 { mag: 16644500, sign: true });
    data.append(FP8x23 { mag: 12765038, sign: true });
    data.append(FP8x23 { mag: 3468800, sign: false });
    data.append(FP8x23 { mag: 22416828, sign: false });
    data.append(FP8x23 { mag: 9311262, sign: true });
    data.append(FP8x23 { mag: 4992094, sign: false });
    data.append(FP8x23 { mag: 4625602, sign: false });
    data.append(FP8x23 { mag: 250440, sign: true });
    data.append(FP8x23 { mag: 6905037, sign: false });
    data.append(FP8x23 { mag: 8044989, sign: false });
    data.append(FP8x23 { mag: 19021394, sign: true });
    data.append(FP8x23 { mag: 7640046, sign: false });
    data.append(FP8x23 { mag: 18743676, sign: true });
    data.append(FP8x23 { mag: 1949779, sign: true });
    data.append(FP8x23 { mag: 8228978, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
