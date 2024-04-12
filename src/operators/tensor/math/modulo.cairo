use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::numbers::NumberTrait;

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
    impl TPartialOrd: PartialOrd<T>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    mut a: Tensor<T>, mut b: Tensor<T>, fmod: Option<bool> 
) -> Tensor<T> {
    match fmod {
            Option::Some(value) => { 

                if value == true {
                return fmod(x, y);
                }
                else if value == false {
                return modop(x, y);
                } 
                else {
                core::panic_with_felt252('invalid fmod') 
                }
                
                },
            Option::None => { return modop(x, y); }
        }
}


fn fmodulo<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TAdd: Add<T>,
    impl TSub: Sub<T>,
    impl TMul: Mul<T>,
    impl TDiv: Div<T>,
    impl TTensorAdd: Add<Tensor<T>>,
    impl TPartialOrd: PartialOrd<T>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>( x:Tensor<T> , y:Tensor<T> ) -> Tensor<T> {

    
    let xx = x.abs();
    let yy = y.abs();

    let mut vals =  (xx / yy);

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


    let remainder = xx - flr * yy;

    let result = remainder * x.sign();

    return result;
}



fn modop(mut x:Tensor<T> , mut y:Tensor<T> ) -> Tensor<T> {


    let mut vals =  (x / y);

    let mut data_result = Array<T> = array![];

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


    let result = x - flr * y;

    return result;
}

