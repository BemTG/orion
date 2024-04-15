use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::numbers::NumberTrait;
use core::array::{ArrayTrait, SpanTrait};
use orion::numbers::fixed_point::core::FixedTrait;


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

    if fmod.unwrap() == true {
         let x = self.abs();
         let b = b.abs();
    } else {
        let x = self.clone();
        let b = b.clone();
    }
    let mut vals =  x / b;

    let mut data_result : Array<T> = array![];

    loop {
        match vals.data.pop_front() {  
            Option::Some(item) => {
                let mut temp = FixedTrait::<T, MAG>::floor(*item);
                data_result.append(temp);
            },
            Option::None(_) => {
                break;
            }
        };
    };

    let flr = TensorTrait::<T, MAG>::new(x.shape, data_result.span());


    let mut result = x - flr * b;

    if fmod.unwrap() == true {

        result = result * x.sign();


    }  

    return result;
}