use core::debug::PrintTrait;
use orion::numbers::NumberTrait;
use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor};
use core::traits::Into;
use orion::numbers::{U32IntoI32, I32IntoU32, I32Div, I32Number};
use orion::operators::nn::core::NNTrait;
use orion::operators::nn::functional;
use orion::operators::vec::{NullableVec, NullableVecImpl};
use orion::operators::tensor::core::{stride};


#[derive(Copy, Drop)]
enum ACTIVATIONS {
    SIGMOID,
    TANH,

}

#[derive(Copy, Drop)]
enum DIRECTION {
    FORWARD,
    REVERSE,
    BIDIRECTIONAL,
}


fn rnn<
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
    +Sub<Tensor<T>>,
    +Mul<Tensor<T>>,
>(
     X: @Tensor<T>,
     R: @Tensor<T>,
     W: @Tensor<T>,
     B: Option<Tensor<T>>,
     sequence_length: Option<Tensor<T>>,
     initial_h: Option<Tensor<T>>,
     activation_alpha: Option<Array<Tensor<T>>>,
     activation_beta: Option<Array<Tensor<T>>>,
     activations: Option<ACTIVATIONS>,
     clip: Option<T>,
     direction: Option<DIRECTION>,
     hidden_size: Option<usize>,
     layout: Option<usize>,
     linear_before_reset: Option<usize>,
     n_outputs: Option<usize>
) -> Array<Tensor<T>> {


    let num_directions = *(*W).shape[0];
    let batch_size = *(*X).shape[1];

    let mut X = X;
    let mut W = W;
    let mut R = R;
    let mut B = B;
    let mut sequence_length = sequence_length;
    let mut initial_h = initial_h;
    let mut hidden_size = hidden_size;
    let mut layout = layout;
    let mut linear_before_reset = linear_before_reset;    

    if num_directions == NumberTrait::<usize>::one() {
        R = @R.squeeze(axes: Option::None(()));
        W = @W.squeeze(axes: Option::None(()));

        if B.is_some()  {
            B = Option::Some(B.unwrap().squeeze(axes: Option::None(())))
        };

        if sequence_length.is_some()  {
            sequence_length = Option::Some(sequence_length.unwrap().squeeze(axes: Option::None(())))
        };

        if initial_h.is_some()  {
           initial_h =  Option::Some(initial_h.unwrap().squeeze(axes: Option::None(())))
        };

        hidden_size = Option::Some( *(*R).shape[ (*R).shape.len() - 1 ] ); 

        if layout.is_none() || layout.unwrap() == NumberTrait::<usize>::zero() {
            X = X
        } else {
            X = @TensorTrait::<T>::transpose(X, array![1, 0, 2].span())
        };

        if B.is_none() {
            let mut b_data_vals = array![];
            let b_data_len = 2 * hidden_size.unwrap();
            let mut i = 0;
            while i < b_data_len {
                b_data_vals.append(NumberTrait::<T>::zero());
                i += 1;
            };

            B = Option::Some(TensorTrait::<T>::new(
                shape: array![2 *  hidden_size.unwrap()].span(),
                data: b_data_vals.span()
            ))
        };


        if initial_h.is_none() {
         
            let mut h_data_vals = array![];
            let h_data_len = batch_size * hidden_size.unwrap();
            let mut i = 0;
            while i < h_data_len {
                h_data_vals.append(NumberTrait::<T>::zero());
                i += 1;
            };

            initial_h = Option::Some(TensorTrait::<T>::new(
                shape: array![ 1, batch_size, hidden_size.unwrap()].span(),
                data: h_data_vals.span()
            ))
        };
  
    }else{
        core::panic_with_felt252('Unsupported num_directions') 
    }

    let result = step(X, R,  W, @B.unwrap(), @initial_h.unwrap(), num_directions, linear_before_reset, layout);

    if n_outputs.unwrap() == NumberTrait::<usize>::one() {
        return array![*result[0]];
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
    +Sub<Tensor<T>>,
    +Mul<Tensor<T>>,
>(
    X: @Tensor<T>,
    R: @Tensor<T>,
    W: @Tensor<T>,
    B: @Tensor<T>,
    H_0: @Tensor<T>,
    num_directions: usize,
    linear_before_reset: Option<usize>,
    layout: Option<usize>,
) -> Array<Tensor<T>> {

    let seq_length = *(*X).shape[0];
    // let rank = (*X).shape.len();
    let hidden_size = *(*H_0).shape[ (*H_0).shape.len() - 1 ] ;
    let batch_size = *(*X).shape[1]; 

    let mut y_data_vals = array![];
    let y_data_vals_len = seq_length * num_directions * batch_size * hidden_size;
    let mut i = 0;
    while i < y_data_vals_len {
        y_data_vals.append(NumberTrait::<T>::zero());
        i += 1;
    };

    let mut Y = TensorTrait::<T>::new(
        shape: array![seq_length, num_directions, batch_size, hidden_size].span(),
        data: y_data_vals.span()
    );

    let (mut b_i, mut b_o) = {
            let mut b_split = split_tensor(B, 2, 0);
            (*b_split[0], *b_split[1])
        };


    let mut h_list: Array<Tensor<T>> = array![];        

    let w_transposed = W.transpose(axes: reverse_axes(*W.shape));
    let r_transposed = R.transpose(axes: reverse_axes(*R.shape));

    let mut H_t = H_0;
    let mut H = H_0;
 
    let X_segment = split_tensor(X, *(*X).shape[0], 0);
    let mut z = 0;
    while z < (X_segment).len() {

        let mut C1 = ( (X_segment[z].unsqueeze(axes: array![0].span()).matmul(@w_transposed) ) ) ;
        let mut C2 =( ( H_t.matmul(@r_transposed)) );

        H = @f_tanh (
            @( (X_segment[z].unsqueeze(axes: array![0].span()).matmul(@w_transposed) )
            + ( ( H_t.matmul(@r_transposed))) 
            + (b_i + b_o) )
        );

        h_list.append(*H);
        H_t = H;
        z += 1;
    };


    let mut concatenated_tensor_h = if h_list.len() > 1 {
        concat_tensors_in_array(h_list)
    } else {
        *h_list[0]
    };
    let mut output: Array<Tensor<T>> = array![];

    if num_directions == 1 {
        Y = concatenated_tensor_h.reshape(
            array![(*Y.shape[0]).into(), (*Y.shape[1]).into(), 
                   (*Y.shape[2]).into(), (*Y.shape[3]).into()].span(),
            false
        );
    }

    if layout.is_some() && layout.unwrap() == 0 || layout.is_none() {
        let mut Y_h = Y.slice(
            starts: array![*Y.shape[0] - 1, 0, 0, 0].span(),
            ends: array![*Y.shape[0], *Y.shape[1], *Y.shape[2], *Y.shape[3]].span(),
            axes: Option::Some(array![0, 1, 2, 3].span()),
            steps: Option::None(())
        );
    
        Y_h = Y_h.squeeze(axes: Option::Some(array![0].span())); 
        output.append(Y);
        output.append(Y_h);
    } else {
        Y = Y.transpose(axes: array![2, 0, 1, 3].span());
        let mut Y_h = Y.slice(
            starts: array![0, 0, *Y.shape[2] - 1, 0].span(),
            ends: array![*Y.shape[0], *Y.shape[1], *Y.shape[2], *Y.shape[3]].span(),
            axes: Option::Some(array![0, 1, 2, 3].span()),
            steps: Option::None(())
        );
        
        Y_h = Y_h.squeeze(axes: Option::Some(array![2].span()));
        output.append(Y);
        output.append(Y_h);
    }

    output
}

fn concat_tensors_in_array<T, MAG, +TensorTrait<T>, +NumberTrait<T, MAG>, +PrintTrait<T>, +Copy<T>, +Drop<T>>(
    tensor_list: Array<Tensor<T>>
) -> Tensor<T> {
    if tensor_list.len() == 1 {
        return *tensor_list[0];
    }

    let mut concatenated_tensor = *tensor_list[0];
    let mut i = 1;
    while tensor_list.len() > i {
        concatenated_tensor = TensorTrait::concat(
            tensors: array![concatenated_tensor, *tensor_list[i]].span(),
            axis: 0,
        );
        i += 1;
    };

    concatenated_tensor
}


fn f_tanh<
    T,
    MAG,
    +TensorTrait<T>,
    +NumberTrait<T, MAG>,
    +Copy<T>,
    +Drop<T>,
    +PrintTrait<T>,
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
    +PrintTrait<T>,
>(
    mut tensor: @Tensor<T>,
    num_outputs: usize,
    mut axis: usize,
) -> Array<Tensor<T>> {

    if (*tensor).shape.len() < 2  {
        tensor = @TensorTrait::<T>::new(
            shape: array![1, (*tensor).data.len()].span(),
            data: *tensor.data
        );
    axis =   1;
    };

    let dim_size = *(*tensor).shape[axis];   

    assert!(dim_size % num_outputs == 0, "Dimension size must be divisible");

    let slice_size = dim_size / num_outputs;
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
                ends.append(*(*tensor).shape[i]);   
            }
            i += 1;
        };

        let slice = tensor.slice(starts.span(), ends.span(), Option::None(()), Option::None(()));

        slices.append(slice.squeeze(axes: Option::None(())));
        start += slice_size;
    };

    slices
}