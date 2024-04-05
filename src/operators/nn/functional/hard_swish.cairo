use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};

fn hard_swish<
    T,
    MAG,
    impl TNumber: NumberTrait<T, MAG>,
    impl TTensor: TensorTrait<T>,
    impl TPartialOrd: PartialOrd<T>,
    impl TAdd: Add<T>,
    impl TMul: Mul<T>,
    impl TDiv: Div<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
>(
    mut x: Tensor<T>
) -> Tensor<T> {
    let mut data_result: Array<T> = array![];

    let a:usize = 6;
    let alpha = NumberTrait::<T, MAG>::one() / NumberTrait::<T, MAG>::unscaled(a.into(), false);
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

    TensorTrait::new(x.shape, data_result.span())
}

