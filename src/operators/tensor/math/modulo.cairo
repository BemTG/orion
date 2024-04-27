use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::numbers::NumberTrait;
use core::array::{ArrayTrait, SpanTrait};
use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::{FP16x16, FP16x16Impl};

use orion::operators::tensor::core::{stride};
use orion::operators::tensor::{FP16x16Tensor, U32Tensor,};
use orion::operators::vec::{NullableVec, NullableVecImpl};
use core::debug::PrintTrait;


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
    +PartialEq<T>,
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

    'check1'.print();
    let mut quotient =  *dividend / *divisor;
     'check2'.print();

    let mut res_data : Array<T> = array![];

    loop {
        match quotient.data.pop_front() {  
            Option::Some(val) => {

                if *val % NumberTrait::<T>::one()  != NumberTrait::<T>::zero() {
                'check3'.print();
                let mut temp = NumberTrait::floor(*val);
                res_data.append(temp);}
            },
            Option::None(_) => {
                break;
            }
        };
    };

    'check4'.print();
    if res_data.len() != 0 {
    let quotient = TensorTrait::<T>::new(*self.shape, res_data.span());
    }

    'check5'.print();

    let mut result = *dividend - quotient * *divisor;

    'check6'.print();

    if fmod.is_some() && fmod.unwrap() == true {

        result = result * dividend.sign();
        'check7'.print();
    }  
    

    return result;
}