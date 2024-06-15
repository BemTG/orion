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
    +Sub<Tensor<T>>,
     +Mul<Tensor<T>>,
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
    direction: Option<DIRECTION>,
     hidden_size: Option<usize>,
     layout: Option<usize>,
     linear_before_reset: Option<usize>,
    n_outputs: Option<usize>
) -> Array<Tensor<T>> {
    let num_directions = *(*W).shape[0];
    let mut h_0 = TensorTrait::<T>::new (array![].span(), array![NumberTrait::<T>::zero()].span());
    let mut H_0 = TensorTrait::<T>::new(array![].span(), array![NumberTrait::<T>::zero()].span());
    let mut b = TensorTrait::<T>::new(array![].span(),  array![NumberTrait::<T>::zero()].span());
    let number_of_gates: usize = 3;

    let mut X = X;
    let mut W = W;
    let mut R = R;
    let mut B = B;
    let mut sequence_length = sequence_length;
    let mut initial_h = initial_h;
    let mut hidden_size = hidden_size;
    let mut layout = layout;
    let mut linear_before_reset = linear_before_reset;

    'checkp1'.print();

    if num_directions == NumberTrait::<usize>::one() {
        'checkp2'.print();
        R = @R.squeeze(axes: Option::None(()));
        W = @W.squeeze(axes: Option::None(()));

        'checkp3'.print();
        if B.is_some() {
            B = Option::Some(B.unwrap().squeeze(axes: Option::None(())))
        };

        if sequence_length.is_some() {
            sequence_length = Option::Some(sequence_length.unwrap().squeeze(axes: Option::None(())))
        };

        if initial_h.is_some() {
           initial_h =  Option::Some(initial_h.unwrap().squeeze(axes: Option::None(())))
        };

        'checkp4'.print();

        hidden_size = Option::Some( *(*R).shape[ (*R).shape.len() - 1 ] ); 
        let batch_size = *(*X).shape[1];

        'checkp5'.print();

        if layout.is_none() || layout.unwrap() == NumberTrait::<usize>::zero() {
            X = X
        } else {
            'the x swapped inside'.print();
            X = @TensorTrait::<T>::transpose(X, array![1, 0, 2].span())
        };

        'checkp6'.print();

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

        'checkp7'.print();

        if initial_h.is_some() {
            h_0 = initial_h.unwrap()
        } else {
            let mut h_data_vals = array![];
            let h_data_len = batch_size * hidden_size.unwrap();
            let mut i = 0;
            while i < h_data_len {
                h_data_vals.append(NumberTrait::<T>::zero());
                i += 1;
            };

            h_0 = TensorTrait::<T>::new(
                shape: array![batch_size, hidden_size.unwrap()].span(),
                data: h_data_vals.span()
            )
        };

        'checkp8'.print();

        B = Option::Some(b);
        H_0 = h_0;
    }else{
        core::panic_with_felt252('Unsupported num_directions') 
    }

    'checkp9'.print();

    let result = step(X, W, R, @B.unwrap(), @H_0, num_directions, linear_before_reset, layout);

    'checkp10'.print();

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
    W: @Tensor<T>,
    R: @Tensor<T>,
    B: @Tensor<T>,
    H_0: @Tensor<T>,
    num_directions: usize,
    linear_before_reset: Option<usize>,
    layout: Option<usize>,
) -> Array<Tensor<T>> {

    'checkp11'.print();
    let seq_length = *(*X).shape[0];
    let rank = (*X).shape.len();
    let hidden_size = *(*H_0).shape[ (*H_0).shape.len() - 1 ] ;
    let batch_size = *(*X).shape[1]; 

    'checkp12'.print();

    let mut y_data_vals = array![];
    let y_data_vals_len = seq_length * num_directions * batch_size * hidden_size;
    let mut i = 0;
    while i < y_data_vals_len {
        y_data_vals.append(NumberTrait::<T>::zero());
        i += 1;
    };

    'checkp13'.print();

    let mut Y = TensorTrait::<T>::new(
        shape: array![seq_length, num_directions, batch_size, hidden_size].span(),
        data: y_data_vals.span()
    );

    'checkp14'.print();

    let mut h_list: Array<Tensor<T>> = array![];

    let ( mut w_z,mut  w_r, mut w_h) = {
        let w_split = split_tensor(W, 3, 0);
        (*w_split[0], *w_split[1], *w_split[2])
    };

    'checkp14b'.print();
    
    let (mut r_z, mut r_r, mut r_h) = {
        let r_split = split_tensor(R, 3, 0);
        (*r_split[0], *r_split[1], *r_split[2])
    };

    let (mut w_bz, mut w_br, mut w_bh, mut r_bz, mut r_br, mut r_bh) = {
        let b_split = split_tensor(B, 6, 0);
        (*b_split[0], *b_split[1], *b_split[2],
         *b_split[3], *b_split[4], *b_split[5])
    };

    'checkp15'.print();

    let gates_w = TensorTrait::<T>::concat(tensors: array![w_z, w_r].span(), axis: 0);
    let gates_r = TensorTrait::<T>::concat(tensors: array![r_z, r_r].span(), axis: 0);
    let gates_b1 = TensorTrait::<T>::concat(tensors: array![w_bz, w_br].span(), axis: 0);
    let gates_b2 = TensorTrait::<T>::concat(tensors: array![r_bz, r_br].span(), axis: 0);
    let gates_b = gates_b1 + gates_b2;

    'checkp16'.print();

    let gates_w_transposed = gates_w.transpose(axes: reverse_axes(gates_w.shape));
    let gates_r_transposed = gates_r.transpose(axes: reverse_axes(gates_r.shape));

    'checkp17'.print();

    let mut H_t = H_0;
    let mut H = H_0;

    'checkp18'.print();
    'the x swapped outside'.print();
    (*(*X).shape[0]).print();

    let X_segment = split_tensor(X, *(*X).shape[0], 0);
    'checkp19'.print();
    (X_segment).len().print();
    let mut i = 0;
    while i < (X_segment).len() {
        'checkp20'.print();
        let gates = (X_segment[i].unsqueeze(axes: array![0].span()).matmul(@gates_w_transposed)
            + H_t.matmul(@gates_r_transposed).unsqueeze(axes: array![0].span())
            + gates_b);


        'checkp21'.print();
        let (mut z, mut r) = {
            let gates_split = split_tensor(@gates, 2, gates.shape.len() - 1);
            (*gates_split[0], *gates_split[1])
        };

        // 'the z shape'.print();

        // (z).shape.len().print();
        // (*(z).shape[0]).print();
        // (*(z).shape[1]).print();

        // 'the z shape'.print();

        // ((z).data.len()).print();
        // (*(z).data[0]).print();
        // (*(z).data[1]).print();
        // (*(z).data[2]).print();

        'checkp22'.print();
        z = f(z);
        'checkp22aa'.print();
        r = f(r);
        'checkp22bb'.print();
        
        'checkp23'.print();
        let w_h_tranposed = w_h.transpose(axes: reverse_axes(w_h.shape));
        let r_h_tranposed = r_h.transpose(axes: reverse_axes(r_h.shape));

        'checkp24'.print();


        let mut h_default = X_segment[i].matmul( @w_h_tranposed )  + (r * *H_t).matmul( @r_h_tranposed ) + w_bh + r_bh;
        h_default = g(@h_default);

        'checkp25aa'.print();
        let mut h_linear = X_segment[i].matmul(@w_h_tranposed) + (r * (H_t.matmul(@r_h_tranposed) + r_bh)) + w_bh;
        h_linear = g( @h_linear);

        'checkp25'.print();

        let mut h = if linear_before_reset.is_some() && linear_before_reset.unwrap() == 0 || linear_before_reset.is_none() {
            h_linear
        } else {
            h_default
        };

        'checkp26'.print();

        let one = TensorTrait::<T>::new(
            shape: array![].span(),
            data: array![NumberTrait::<T>::one()].span(),
        );

        'checkp27'.print();

        H =  @(((one - z) * h) + (z * *H_t));

        'checkp28'.print();

        h_list.append(*H);
        H_t = H;
        i += 1;
    };

    'checkp29'.print();
 
    let mut concatenated = if h_list.len() > 1 {
        concat_tensors_in_array(h_list)
    } else {
        *h_list[0]
    };

    'checkp30'.print();

    let mut output: Array<Tensor<T>> = array![];

    if num_directions == 1 {
        // Y = concatenated.reshape(
        //     array![(*Y.shape[0]).into(), (*Y.shape[1]).into(), 
        //            (*Y.shape[2]).into(), (*Y.shape[3]).into()].span(),
        //     false
        // );

        let concatenated_h_list_tensors = Option::Some(concatenated);

        let Y_strides = stride(Y.shape);


        let mut Y_data = NullableVecImpl::<T>::new(); // converting Y values to nullable vec
        let mut i = 0;
        while i != Y.data.len() {
            Y_data.push(*Y.data.at(i));
            i += 1;
        };

        let process = match concatenated_h_list_tensors {
        Option::Some(item) => {
        let mut i = 0;
        while i != *Y.shape.at(0) {
            let mut j = 0;
            while j != *Y.shape.at(2) {
                let mut k = 0;
                while k != *Y.shape.at(3) {
                    let concatenate_val = item.at(array![ 0, j, k].span());
                    let y_offset = i * *Y_strides.at(0) + 0 * *Y_strides.at(1) + j * *Y_strides.at(2) + k;
                    Y_data.set(y_offset, concatenate_val);
                    k += 1;
                };
                j += 1;
            };
            i += 1;
        }
    },
    Option::None => {},
};

let mut res_data: Array<T> = array![];
let mut i = 0;
while i != Y_data.len() {
    res_data.append(Y_data.at(i));
    i += 1;
};

Y = TensorTrait::new(Y.shape, res_data.span());

        output.append(Y);
    }

    'checkp31'.print();

    if layout.is_some() && layout.unwrap() == 0 || layout.is_none() {
        let mut Y_h = Y.slice(
            starts: array![*Y.shape[0] - 1, 0, 0, 0].span(),
            ends: array![*Y.shape[0], *Y.shape[1], *Y.shape[2], *Y.shape[3]].span(),
            axes: Option::Some(array![0, 1, 2, 3].span()),
            steps: Option::None(())
        );
    
        Y_h = Y_h.squeeze(axes: Option::Some(array![0].span())); 
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
        output.append(Y_h);
    }

    output
}

fn concat_tensors_in_array<T, MAG, +TensorTrait<T>, +NumberTrait<T, MAG>, +Copy<T>, +Drop<T>>(
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
    +PrintTrait<T>,
>(
    mut z: Tensor<T>
) -> Tensor<T> {
    let mut data_result: Array<T> = array![];

    loop {
        match z.data.pop_front() {
            Option::Some(item) => {
                'the item'.print();
                (*item).print();
                let result = NumberTrait::one()
                    / (NumberTrait::one() + (*item * NumberTrait::neg_one()).exp());
                'the result'.print();
                result.print();
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

    if (*tensor).shape.len() < 2  {
        tensor = @TensorTrait::<T>::new(
            shape: array![1, (*tensor).data.len()].span(),
            data: *tensor.data
        );
    axis =   1;
    };

    

    
    let dim_size = *(*tensor).shape[axis];   

    'dimsize'.print();
    (dim_size).print();
    'numoutput'.print();
    num_outputs.print();

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