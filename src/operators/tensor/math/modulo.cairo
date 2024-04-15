use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::numbers::NumberTrait;
use core::array::{ArrayTrait, SpanTrait};
use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::{FP16x16, FP16x16Impl};

use orion::operators::tensor::core::{stride};
use orion::operators::tensor::{FP16x16Tensor, U32Tensor,};
use orion::operators::vec::{NullableVec, NullableVecImpl};


/// Cf: TensorTrait::Mod docstring


fn modulo<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TTensorAdd: Add<Tensor<T>>,
    impl TTensorSub: Sub<Tensor<T>>,
    impl TTensorDiv: Div<Tensor<T>>,
    impl TTensorMul: Mul<Tensor<T>>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
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

    let mut quotient =  *dividend / *divisor;

    let mut res_data : Array<T> = array![];

    loop {
        match quotient.data.pop_front() {  
            Option::Some(val) => {
                let mut temp = NumberTrait::floor(*val);
                res_data.append(temp);
            },
            Option::None(_) => {
                break;
            }
        };
    };

    let floored_quotients = TensorTrait::<T>::new(*self.shape, data_result.span());


    let mut result = *dividend - floored_quotients * *divisor;

    if fmod.is_some() && fmod.unwrap() == true {

        result = result * dividend.sign();


    }  

    result;
}