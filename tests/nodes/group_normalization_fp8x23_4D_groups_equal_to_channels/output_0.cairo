use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 16041730, sign: true });
    data.append(FP8x23 { mag: 10947809, sign: true });
    data.append(FP8x23 { mag: 13189602, sign: true });
    data.append(FP8x23 { mag: 12412579, sign: true });
    data.append(FP8x23 { mag: 16374650, sign: false });
    data.append(FP8x23 { mag: 14252273, sign: false });
    data.append(FP8x23 { mag: 11784024, sign: false });
    data.append(FP8x23 { mag: 13376401, sign: false });
    data.append(FP8x23 { mag: 14906940, sign: true });
    data.append(FP8x23 { mag: 11217985, sign: true });
    data.append(FP8x23 { mag: 11396898, sign: true });
    data.append(FP8x23 { mag: 15069897, sign: true });
    data.append(FP8x23 { mag: 13807433, sign: false });
    data.append(FP8x23 { mag: 11371682, sign: false });
    data.append(FP8x23 { mag: 15383500, sign: false });
    data.append(FP8x23 { mag: 15224733, sign: false });
    data.append(FP8x23 { mag: 13164436, sign: true });
    data.append(FP8x23 { mag: 12725046, sign: true });
    data.append(FP8x23 { mag: 16134860, sign: true });
    data.append(FP8x23 { mag: 10567378, sign: true });
    data.append(FP8x23 { mag: 11238824, sign: false });
    data.append(FP8x23 { mag: 13473113, sign: false });
    data.append(FP8x23 { mag: 15709862, sign: false });
    data.append(FP8x23 { mag: 15365549, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
