use array::ArrayTrait;
use option::OptionTrait;
use array::SpanTrait;

use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait, unravel_index};
use orion::operators::tensor::helpers::{
    broadcast_shape, broadcast_index_mapping, len_from_shape, check_compatibility
};

/// Cf: TensorTrait::and docstring
fn and<
    T,
    MAG,
    impl TNumber: NumberTrait<T, MAG>,
    impl UsizeFTensor: TensorTrait<bool>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    y: @Tensor<T>, z: @Tensor<T>
) -> Tensor<bool> {
    let broadcasted_shape = broadcast_shape(*y.shape, *z.shape);
    let mut result: Array<bool> = ArrayTrait::new();

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    loop {
        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*y.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*z.shape, indices_broadcasted);

        if NumberTrait::and(*(*y.data)[indices_self], *(*z.data)[indices_other]) {
            result.append(true);
        } else {
            result.append(false);
        }

        n += 1;
        if n == num_elements {
            break ();
        };
    };

    return TensorTrait::new(broadcasted_shape, result.span());
}
