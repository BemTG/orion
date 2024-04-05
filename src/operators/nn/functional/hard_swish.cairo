use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::{NumberTrait, I32IntoU32, U32IntoI32};
use orion::operators::tensor::core::{Tensor, TensorTrait};

use core::option::OptionTrait;
use core::traits::TryInto;
use orion::numbers::{FP16x16, FP16x16Impl, FP32x32, FP32x32Impl};
use orion::operators::tensor::{
    I8Tensor, I32Tensor, U32Tensor, FP16x16Tensor, BoolTensor
};
use orion::operators::vec::{VecTrait, NullableVec, NullableVecImpl};

fn hard_swish<
    T,
    MAG,
    +NumberTrait<T, MAG>,
    +TensorTrait<T>,
    +PartialOrd<T>,
    +Add<T>,
    +Mul<T>,
    +Div<T>,
    +Copy<T>,
    +Drop<T>,
    +Mul<Tensor<T>>,
    +Into<usize, MAG>,
    >(
    mut x: Tensor<T>
) -> Tensor<T> {
    let x_cloned = x.clone();
    let mut data_result: Array<T> = array![];

    let a:usize = 6;
    let alpha = NumberTrait::<T, MAG>::one() / NumberTrait::<T, MAG>::new_unscaled(a.into(), false);
    let beta = NumberTrait::<T, MAG>::half();

    loop {
        match x.data.pop_front() {
            Option::Some(item) => {
                let temp = (*item) * alpha + beta;
                let result = temp.min(NumberTrait::one()).max(NumberTrait::zero());
                data_result.append(result);
            },
            Option::None => { break; }
        };
    };


     let result = TensorTrait::new(x.shape, data_result.span()).into() * TensorTrait::new(x.shape, data_result.span()).into() ;

     return result;

}

