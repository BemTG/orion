use core::debug::PrintTrait;

use orion::numbers::NumberTrait;

/// Cf: NNTrait::gru docstring


use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
use orion::numbers::{FP16x16, FP16x16Impl, FP8x23};
// use orion::operators::tensor::FP16x16TensorPartialEq;
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::FP16x16Tensor;
use orion::operators::tensor::FP8x23Tensor;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::core::{unravel_index};

use orion::operators::tensor::implementations::tensor_i32::{
    I32Tensor, I32TensorAdd, I32TensorSub, I32TensorMul, I32TensorDiv, I32TensorPartialEq,
    TensorI8IntoTensorI32
};

use core::traits::TryInto;
use core::option::OptionTrait;
use core::traits::Into;


use orion::numbers::fixed_point::implementations::fp16x16::math::lut;
use core::integer;

use orion::operators::tensor::implementations::tensor_fp16x16::{
    FP16x16TensorAdd, FP16x16TensorSub, FP16x16TensorMul, FP16x16TensorDiv, FP16x16TensorPartialEq,
};


use orion::operators::tensor::implementations::tensor_u32::{
    U32TensorAdd, U32TensorSub, U32TensorMul, U32TensorDiv, U32TensorPartialEq,
};

use orion::operators::nn::{FP8x23NN, FP16x16NN};
use orion::numbers::{U32IntoI32, I32IntoU32, I32Div, I32Number};

use orion::numbers::fixed_point::core::FixedTrait;

use orion::numbers::{FP32x32, FP32x32Impl};

use orion::operators::vec::{VecTrait, NullableVec, NullableVecImpl};
use orion::operators::tensor::helpers::{reduce_output_shape, len_from_shape, combine_indices};
use orion::operators::tensor::math::{reduce_sum::accumulate_sum, arithmetic::div_downcast};

use orion::operators::tensor::{math, linalg, quantization, core as core_tensor, ml, manipulation};

use orion::numbers::fixed_point::implementations::fp16x16::core::{
    HALF, ONE, MAX, FP16x16Add, FP16x16AddEq, FP16x16Sub, FP16x16Mul, FP16x16MulEq,
    FP16x16TryIntoU128, FP16x16PartialEq, FP16x16PartialOrd, FP16x16SubEq, FP16x16Neg, FP16x16Div,
    FP16x16IntoFelt252,
};

use orion::operators::nn::core::NNTrait;
use orion::operators::nn::functional;


use core::serde::Serde;

use alexandria_data_structures::array_ext::{SpanTraitExt};

use orion::operators::tensor::helpers::{check_shape, broadcast_index_mapping};


use alexandria_data_structures::array_ext::ArrayTraitExt;

use orion::operators::tensor::{core::{stride}, BoolTensor};

use orion::operators::tensor::helpers::{broadcast_shape,};

#[derive(Copy, Drop)]
enum ACTIVATIONS {
    SIGMOID,
    TANH,

}

#[derive(Copy, Drop)]
enum DIRECTIONS {
    FORWARD,
    REVERSE,
    BIDIRECTIONAL,
}


fn gru<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +Copy<T>,
    +Drop<T>,
    +Add<T>,
    +Mul<T>,
    +Sub<T>,
    +Div<T>,
    +AddEq<T>,
    +PrintTrait<T>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +TryInto<T, usize>,
    +Into<usize, MAG>,
    +Rem<T>,
    +Neg<T>,
    +SubEq<T>,
    +Add<Tensor<T>>,
>(
     X: @Tensor<T>,
     W: @Tensor<T>,
     R: @Tensor<T>,
     B: Option<Tensor<T>>,
     sequence_length: Option<Tensor<T>>,
    initial_h: Option<Tensor<T>>,
    activation_alpha: Option<Array<Tensor<T>>>,
    activation_beta: Option<Array<Tensor<T>>>,
    activations: Option<ACTIVATIONS>,
    clip: Option<T>,
    direction: Option<DIRECTIONS>,
     hidden_size: Option<usize>,
     layout: Option<usize>,
     linear_before_reset: Option<usize>,
    n_outputs: Option<usize>
) -> Array<Tensor<T>> {
    let num_directions = (*W.shape).at(0);
    let mut h_0 = TensorTrait::<T>::new(shape: array![].span(), data: array![NumberTrait::<T>::zero()].span());
    let mut H_0 = TensorTrait::<T>::new(shape: array![].span(), data: array![NumberTrait::<T>::zero()].span());
    let mut b = TensorTrait::<T>::new(shape: array![].span(), data: array![NumberTrait::<T>::zero()].span());
    let number_of_gates: usize = 3;

    let mut X = X;
    let mut W = W;
    let mut R = R;
    let mut B = B;
    let mut sequence_length = sequence_length;
    let mut initial_h = initial_h;
    // let mut hidden_size = hidden_size;
    let mut layout = layout;
    let mut linear_before_reset = linear_before_reset;

    if *num_directions == NumberTrait::<usize>::one() {
        let R = R.squeeze(axes: Option::None(()));
        let W = W.squeeze(axes: Option::None(()));

        if B.is_some() {
            B = Option::Some(B.unwrap().squeeze(axes: Option::None(())))
        };

        if sequence_length.is_some() {
            sequence_length = Option::Some(sequence_length.unwrap().squeeze(axes: Option::None(())))
        };

        if initial_h.is_some() {
           initial_h =  Option::Some(initial_h.unwrap().squeeze(axes: Option::None(())))
        };

        let hidden_size = Option::Some(*R.shape.at(R.shape.len() - 1));
        let batch_size = (*X.shape).at(1);

        if layout.is_none() || layout.unwrap() == NumberTrait::<usize>::zero() {
            X = X
        } else {
            X = @(TensorTrait::<T>::transpose(X, array![1, 0, 2].span()))
        };

        if B.is_some() {
            b = B.unwrap();
        } else {
            let mut b_data_vals = array![];
            let b_data_len = 2 * number_of_gates * hidden_size.unwrap();
            let mut i = 0;
            while i < b_data_len {
                b_data_vals.append(NumberTrait::<T>::zero());
                i += 1;
            };

            b = TensorTrait::<T>::new(
                shape: array![2 * number_of_gates * hidden_size.unwrap()].span(),
                data: b_data_vals.span()
            )
        };

        if initial_h.is_some() {
             h_0 = initial_h.unwrap()
        } else {
            let mut h_data_vals = array![];
            let h_data_len = *batch_size * hidden_size.unwrap();
            let mut i = 0;
            while i < h_data_len {
                h_data_vals.append(NumberTrait::<T>::zero());
                i += 1;
            };

             h_0 = TensorTrait::<T>::new(
                shape: array![*batch_size, hidden_size.unwrap()].span(),
                data: h_data_vals.span()
            )
        };

        B = Option::Some(b);
        H_0 = h_0;
    }else{
        core::panic_with_felt252('Unsupported value') 
    }

    let result = step(X, W, R, @B.unwrap(), @H_0, *num_directions, linear_before_reset, layout);

    if n_outputs.unwrap() == NumberTrait::<usize>::one() {
        return array![*result.at(0)];
    } else {
        return result;
    }
}

fn step<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +Copy<T>,
    +Drop<T>,
    +Add<T>,
    +Mul<T>,
    +Sub<T>,
    +Div<T>,
    +AddEq<T>,
    +PrintTrait<T>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +TryInto<T, usize>,
    +Into<usize, MAG>,
    +Rem<T>,
    +Neg<T>,
    +SubEq<T>,
    +Add<Tensor<T>>,
    // +Mul<Tensor<T>>,
>(
    X: @Tensor<T>,
    W: @Tensor<T>,
    R: @Tensor<T>,
    B: @Tensor<T>,
    H_0: @Tensor<T>,
    num_directions: usize,
    linear_before_reset: Option<usize>,
    layout: Option<usize>,
) -> Array<Tensor<T>> {
    let seq_length = (*X.shape).at(0);
    let rank = (*X.shape).len();
    let hidden_size = (*H_0.shape).at((*H_0.shape).len() - 1);
    let batch_size = (*X.shape).at(1);               

    let mut y_data_vals = array![];
    let y_data_vals_len = *seq_length * num_directions * *batch_size * *hidden_size;
    let mut i = 0;
    while i < y_data_vals_len {
        y_data_vals.append(NumberTrait::<T>::zero());
        i += 1;
    };

    let mut Y = TensorTrait::<T>::new(
        shape: array![*seq_length, num_directions, *batch_size, *hidden_size].span(),
        data: y_data_vals.span()
    );

    let mut h_list: Array<Tensor<T>> = array![];

    let (w_z, w_r, w_h) = {
        let w_split = split_tensor(W, 3, 0);
        (*w_split.at(0), *w_split.at(1), *w_split.at(2))
    };
    
    let (r_z, r_r, r_h) = {
        let r_split = split_tensor(R, 3, 0);
        (*r_split.at(0), *r_split.at(1), *r_split.at(2))
    };

    let (w_bz, w_br, w_bh, r_bz, r_br, r_bh) = {
        let b_split = split_tensor(B, 6, 0);
        (*b_split.at(0), *b_split.at(1), *b_split.at(2),
         *b_split.at(3), *b_split.at(4), *b_split.at(5))
    };

    let gates_w = TensorTrait::<T>::concat(tensors: array![w_z, w_r].span(), axis: 0);
    let gates_r = TensorTrait::<T>::concat(tensors: array![r_z, r_r].span(), axis: 0);
    let gates_b1 = TensorTrait::<T>::concat(tensors: array![w_bz, w_br].span(), axis: 0);
    let gates_b2 = TensorTrait::<T>::concat(tensors: array![r_bz, r_br].span(), axis: 0);
    let gates_b = gates_b1 + gates_b2;

    let gates_w_transposed = gates_w.transpose(axes: reverse_axes(gates_w.shape));
    let gates_r_transposed = gates_r.transpose(axes: reverse_axes(gates_r.shape));

    let mut H_t = H_0;
    let mut H = H_0;

    let X_segment = split_tensor(X, *(*X.shape).at(0), 0);
    let mut i = 0;
    while i < (X_segment).len() {
        let gates = (X_segment.at(i).unsqueeze(axes: array![0].span()).matmul(@gates_w_transposed)
            + H_t.matmul(@gates_r_transposed).unsqueeze(axes: array![0].span())
            + gates_b);

        let (mut z, mut r) = {
            let gates_split = split_tensor(@gates, 2, gates.shape.len() - 1);
            (*gates_split.at(0), *gates_split.at(1))
        };

        z = f(z);
        r = f(r);
        
        let w_h_tranposed = w_h.transpose(axes: reverse_axes(w_h.shape));
        let r_h_tranposed = r_h.transpose(axes: reverse_axes(r_h.shape));

        let h_default = g((@X_segment.at(i).matmul(@w_h_tranposed))
            + ((r * *H_t).matmul(@r_h_tranposed))
            + w_bh + r_bh);

        let h_linear = g((@X_segment.at(i).matmul(@w_h_tranposed))
            + (r * (H_t.matmul(@r_h_tranposed) + r_bh))
            + w_bh);

        let mut h = if linear_before_reset.is_some() && linear_before_reset.unwrap() == 0 || linear_before_reset.is_none() {
            h_linear
        } else {
            h_default
        };

        let one = TensorTrait::<T>::new(
            shape: array![].span(),
            data: array![NumberTrait::<T>::one()].span(),
        );

        H = ((one - z) * h + z * H_t);

        h_list.append(*H);
        H_t = H;
        i += 1;
    };
 
    let concatenated = if h_list.len() > 1 {
        concat_tensor_array(h_list)
    } else {
        *h_list.at(0)
    };

    let mut output: Array<Tensor<T>> = array![];

    if num_directions == 1 {
        Y = concatenated.reshape(
            array![(*Y.shape.at(0)).into(), (*Y.shape.at(1)).into(), 
                   (*Y.shape.at(2)).into(), (*Y.shape.at(3)).into()].span(),
            false
        );

        output.append(Y);
    }

    if layout.is_some() && layout.unwrap() == 0 || layout.is_none() {
        let mut Y_h = Y.slice(
            starts: array![*Y.shape.at(0) - 1, 0, 0, 0].span(),
            ends: array![*Y.shape.at(0), *Y.shape.at(1), *Y.shape.at(2), *Y.shape.at(3)].span(),
            axes: Option::Some(array![0, 1, 2, 3].span()),
            steps: Option::None(())
        );
    
        Y_h = Y_h.squeeze(axes: Option::Some(array![0].span())); 
        output.append(Y_h);
    } else {
        Y = Y.transpose(axes: array![2, 0, 1, 3].span());
        let mut Y_h = Y.slice(
            starts: array![0, 0, *Y.shape.at(2) - 1, 0].span(),
            ends: array![*Y.shape.at(0), *Y.shape.at(1), *Y.shape.at(2), *Y.shape.at(3)].span(),
            axes: Option::Some(array![0, 1, 2, 3].span()),
            steps: Option::None(())
        );
        
        Y_h = Y_h.squeeze(axes: Option::Some(array![2].span()));
        output.append(Y_h);
    }

    output
}

fn concat_tensor_array<T, MAG, +TensorTrait<T>, +NumberTrait<T, MAG>, +Copy<T>, +Drop<T>>(
    tensor_list: Array<Tensor<T>>
) -> Tensor<T> {
    if tensor_list.len() == 1 {
        return *tensor_list.at(0);
    }

    let mut concatenated_tensor = *tensor_list.at(0);
    let mut i = 1;
    while tensor_list.len() > i {
        concatenated_tensor = TensorTrait::concat(
            tensors: array![concatenated_tensor, *tensor_list.at(i)].span(),
            axis: 0,
        );
        i += 1;
    };

    concatenated_tensor
}



fn f<
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
    mut z: Tensor<T>
) -> Tensor<T> {
    let mut data_result: Array<T> = array![];

    loop {
        match z.data.pop_front() {
            Option::Some(item) => {
                let result = NumberTrait::one()
                    / (NumberTrait::one() + (*item * NumberTrait::neg_one()).exp());
                data_result.append(result);
            },
            Option::None => { break; }
        };
    };

    TensorTrait::new(z.shape, data_result.span())
}




fn g<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +Copy<T>,
    +Drop<T>,
>(
    mut x: @Tensor<T>
) -> Tensor<T> {
    x.tanh()
}

fn reverse_axes(mut x: Span<usize>) -> Span<usize> {
    let mut result = array![];
    loop {
        match x.pop_back() {
            Option::Some(item) => { result.append(*item) },
            Option::None => { break; }
        }
    };
    result.span()
}


fn split_tensor<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +Copy<T>,
    +Drop<T>,
    +Rem<T>,
>(
    mut tensor: @Tensor<T>,
    num_outputs: usize,
    mut axis: usize,
) -> Array<Tensor<T>> {
    if (*tensor.shape).len() < NumberTrait::<usize>::one() + NumberTrait::<usize>::one()  {
        tensor = @TensorTrait::<T>::new(
            shape: array![1, (*tensor).data.len()].span(),
            data: *tensor.data
        );

    axis =  NumberTrait::<usize>::one() ;
    };

    let shape = tensor.shape;
    let dim_size = (*tensor.shape).at(axis);   //  (*X.shape).at(0);

    assert!(*dim_size % num_outputs == 0, "Dimension size must be divisible");

    let slice_size = *dim_size / num_outputs;
    let mut slices = ArrayTrait::new();

    let mut start = 0;
    while start < dim_size {
        let mut starts = ArrayTrait::new();
        let mut ends = ArrayTrait::new();

        let mut i = 0;
        while i !=  (*tensor.shape).len()  {
            if i == axis {
                starts.append(start);
                ends.append(start + slice_size);
            } else {
                starts.append(0);
                ends.append((*tensor.shape).at(i));   
            }
            i += 1;
        };

        let slice = tensor.slice(starts.span(), ends.span(), Option::None(()), Option::None(()));

        slices.append(slice.squeeze(axes: Option::None(())));
        start += slice_size;
    };

    slices
}