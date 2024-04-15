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
    impl TAdd: Add<T>,
    impl TSub: Sub<T>,
    impl TMul: Mul<T>,
    impl TDiv: Div<T>,
    impl TTensorAdd: Add<Tensor<T>>,
    impl TTensorSub: Sub<Tensor<T>>,
    impl TTensorDiv: Div<Tensor<T>>,
    impl TTensorMul: Mul<Tensor<T>>,
    impl TPartialOrd: PartialOrd<T>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>( self: @Tensor<T>,  b: @Tensor<T>, fmod: Option<bool> ) ->  Tensor<T> {

    // let mut x = self;
    // let mut b = b;

    match fmod {
            Option::Some(value) => { 

                if value == true {
                let x = @self.abs();
                let b = @b.abs();
                }
                else if value == false {
                let  x = self;
                let  b = b;
                } 
                else {
                core::panic_with_felt252('invalid fmod') 
                }
                
                },
            Option::None => { 
                let  x = self;
                let  b = b; }
        }

    let mut vals =  *x / *b;

    let mut data_result : Array<T> = array![];

    loop {
        match vals.data.pop_front() {  
            Option::Some(item) => {
                let mut temp = NumberTrait::floor(*item);
                data_result.append(temp);
            },
            Option::None(_) => {
                break;
            }
        };
    };

    let flr = TensorTrait::<T>::new(*self.shape, data_result.span());


    let mut result = *x - flr * *b;

    if fmod != Option::None && fmod.unwrap() == true {

        result = result * x.sign();


    }  

    return result;
}