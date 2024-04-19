use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::numbers::NumberTrait;
use core::array::{ArrayTrait, SpanTrait};
use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::{FP16x16, FP16x16Impl};
use orion::numbers::{U32IntoI32, I32IntoU32, I32Div, I32Number};
use orion::operators::tensor::implementations::tensor_i32::{
    I32Tensor, I32TensorAdd, I32TensorSub, I32TensorMul, I32TensorDiv, I32TensorPartialEq,
    TensorI8IntoTensorI32
};

use orion::operators::tensor::core::{stride};
use orion::operators::tensor::{FP16x16Tensor, U32Tensor,};
use orion::operators::vec::{NullableVec, NullableVecImpl};


/// Cf: TensorTrait::Mod docstring
fn modulo<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +Add<Tensor<T>>,
    +Sub<Tensor<T>>,
    +Div<Tensor<T>>,
    +Mul<Tensor<T>>,
    +Copy<T>,
    +Drop<T>,
    +Rem<T>,
>( self: @Tensor<T>,  divisor: @Tensor<T>, fmod: Option<bool> ) ->  Tensor<T> {

    let mut dividend = self;
    let mut divisor = divisor;

    match fmod {
            Option::Some(value) => { 
                if value == true {
                  dividend = @self.abs();
                  divisor = @divisor.abs();
                }
                else if value != false && value != true {
                core::panic_with_felt252('invalid fmod') 
                }         
                },
            Option::None => { 
                dividend = self;
                divisor = divisor;
                 }
        }

    // let mut quotient =  *dividend / *divisor;

    // let mut res_data : Array<T> = array![];

    // // loop {
    // //     match quotient.data.pop_front() {  
    // //         Option::Some(val) => {
    // //             let mut temp = NumberTrait::floor(*val);
    // //             res_data.append(temp);
    // //         },
    // //         Option::None(_) => {
    // //             break;
    // //         }
    // //     };
    // // };

    // let floored_quotients = TensorTrait::<T>::new(*self.shape, res_data.span());

    // let mut result = *dividend - quotient * *divisor;  // floored_quotients

    // if fmod.is_some() && fmod.unwrap() == true {

    //     result = result * dividend.sign();
    // }  

    let mut result =  self ;  // *dividend % *divisor ;

    return result;
}