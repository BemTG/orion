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
>(self: @Tensor<T>, divisor: @Tensor<T>, fmod: Option<bool>) -> Tensor<T> {
    
    let mut dividend = self.clone();
    let mut divisor = divisor.clone();

    'check0'.print();

    match fmod {
        Option::Some(item) => {
            if item == true {
                dividend = self.abs();
                divisor = divisor.abs();
            }
        },
        Option::None => {},
    }

    'check1'.print();

    let mut expanded_new_shape: Array<usize> = array![];
    if dividend.shape.len() != divisor.shape.len() {
    let shape_diff = (dividend.shape.len() - divisor.shape.len()).abs();

    'not_same'.print();
    dividend.shape.len().print();
    divisor.shape.len().print();

    if dividend.shape.len() < divisor.shape.len() {
        let mut i: usize = 0;
        loop {
            if i >= dividend.shape.len() {
                break;
            }
            expanded_new_shape.append(*dividend.shape.at(i));
            i += 1;
        };

        let mut i: usize = 0;
        loop {
            if i >= shape_diff {
                break;
            }
            expanded_new_shape.append(1);
            i += 1;
        };

        dividend = TensorTrait::<T>::new(expanded_new_shape.span(),  dividend.data);
        

    } else {
        let mut i: usize = 0;
        loop {
            if i >= divisor.shape.len() {
                break;
            }
            expanded_new_shape.append(*divisor.shape.at(i ));
            i += 1;
        };

        let mut i: usize = 0;
        loop {
            if i >= shape_diff {
                break;
            }
            expanded_new_shape.append(1);
            i += 1;
        };
        
        divisor = TensorTrait::<T>::new(expanded_new_shape.span(),  divisor.data);

    }

    'fixed'.print();
    dividend.shape.len().print();
    divisor.shape.len().print();

    }

    let mut quotient = dividend / divisor;

    'check quo'.print();

    let mut res_data : Array<T> = array![];
    loop {
        match quotient.data.pop_front() {
            Option::Some(val) => {
                if *val % NumberTrait::<T>::one() != NumberTrait::<T>::zero() {
                    let mut temp = NumberTrait::floor(*val);
                    res_data.append(temp);
                } else {
                    res_data.append(*val);
                }
            },
            Option::None => {
                break;
            }
        }
    };

    'check quo2'.print();

    quotient = TensorTrait::<T>::new(*self.shape, res_data.span());

    'check quo3'.print();

    let mut remainder = dividend - quotient * divisor;

    if fmod.is_some() && fmod.unwrap() == true {
        remainder = remainder * self.sign();
    }  
    
    return remainder;
}