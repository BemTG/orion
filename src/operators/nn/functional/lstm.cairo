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


fn lstm<
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
    initial_c: Option<Tensor<T>>,
     P: Option<Tensor<T>>,
    activation_alpha: Option<Array<Tensor<T>>>,
    activation_beta: Option<Array<Tensor<T>>>,
    activations: Option<ACTIVATIONS>,
    clip: Option<T>,
    direction: Option<DIRECTION>,
     hidden_size: Option<usize>,
      input_forget: Option<usize>,
     layout: Option<usize>,
     linear_before_reset: Option<usize>,
    n_outputs: Option<usize>
) -> Array<Tensor<T>> {
    let num_directions = *(*W).shape[0];
    let number_of_gates: usize = 4;
    let mut number_of_peepholes :usize  = 3;

    let mut X = X;
    let mut W = W;
    let mut R = R;
    let mut B = B;
    let mut P = P;
    let mut sequence_length = sequence_length;
    let mut initial_h = initial_h;
    let mut initial_c = initial_c;
    let mut hidden_size = hidden_size;
    let mut layout = layout;
    let mut linear_before_reset = linear_before_reset;

    'checkp1'.print();

    if num_directions == NumberTrait::<usize>::one() {
        'checkp2'.print();
        R = @R.squeeze(axes: Option::None(()));
        W = @W.squeeze(axes: Option::None(()));

        'checkp3'.print();
        if B.is_some() && B.unwrap().shape.len() > NumberTrait::<usize>::zero()  && B.unwrap().shape.at(0) == @NumberTrait::<usize>::one() {
            B = Option::Some(B.unwrap().squeeze(axes: Option::None(())))
        };

        if sequence_length.is_some() && (sequence_length.unwrap().shape).len() > NumberTrait::<usize>::zero()  && *sequence_length.unwrap().shape.at(0) == NumberTrait::<usize>::one() {
            sequence_length = Option::Some(sequence_length.unwrap().squeeze(axes: Option::None(())))
        };

        if initial_h.is_some() && (initial_h.unwrap().shape).len() > NumberTrait::<usize>::zero()  && *initial_h.unwrap().shape.at(0) == NumberTrait::<usize>::one() {
           initial_h =  Option::Some(initial_h.unwrap().squeeze(axes: Option::None(())))
        };

        if initial_c.is_some() && (initial_c.unwrap().shape).len() > NumberTrait::<usize>::zero()  && *initial_c.unwrap().shape.at(0) == NumberTrait::<usize>::one() {
           initial_c =  Option::Some(initial_c.unwrap().squeeze(axes: Option::None(())))
        };

        if P.is_some() && (P.unwrap().shape).len() > NumberTrait::<usize>::zero()  && P.unwrap().shape.at(0) == @NumberTrait::<usize>::one(){
            B = Option::Some(B.unwrap().squeeze(axes: Option::None(())))
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

        if B.is_none() {
            let mut b_data_vals = array![];
            let b_data_len = 2 * number_of_gates * hidden_size.unwrap();
            let mut i = 0;
            while i < b_data_len {
                b_data_vals.append(NumberTrait::<T>::zero());
                i += 1;
            };

            B = Option::Some(TensorTrait::<T>::new(
                shape: array![2 * number_of_gates * hidden_size.unwrap()].span(),
                data: b_data_vals.span()
            ))
        };

        if P.is_none() {
            let mut p_data_vals = array![];
            let p_data_len = number_of_peepholes * hidden_size.unwrap();
            let mut i = 0;
            while i < p_data_len {
                p_data_vals.append(NumberTrait::<T>::zero());
                i += 1;
            };

            P = Option::Some(TensorTrait::<T>::new(
                shape: array![number_of_peepholes * hidden_size.unwrap()].span(),
                data: p_data_vals.span()
            ))
        };

        'checkp7'.print();

        if initial_h.is_none() {
         
            let mut h_data_vals = array![];
            let h_data_len = batch_size * hidden_size.unwrap();
            let mut i = 0;
            while i < h_data_len {
                h_data_vals.append(NumberTrait::<T>::zero());
                i += 1;
            };

            initial_h = Option::Some(TensorTrait::<T>::new(
                shape: array![batch_size, hidden_size.unwrap()].span(),
                data: h_data_vals.span()
            ))
        };

        if initial_c.is_none() {
         
            let mut c_data_vals = array![];
            let c_data_len = batch_size * hidden_size.unwrap();
            let mut i = 0;
            while i < c_data_len {
                c_data_vals.append(NumberTrait::<T>::zero());
                i += 1;
            };

            initial_c = Option::Some(TensorTrait::<T>::new(
                shape: array![batch_size, hidden_size.unwrap()].span(),
                data: c_data_vals.span()
            ))
        };

        'checkp8'.print();

    
    }else{
        core::panic_with_felt252('Unsupported num_directions') 
    }

    'checkp9'.print();

    let result = step(X, R,  W, @B.unwrap(), @initial_h.unwrap(), @initial_c.unwrap(), @P.unwrap(),  num_directions, linear_before_reset, layout);

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
    C_0: @Tensor<T>,
    P: @Tensor<T>,
    num_directions: usize,
    linear_before_reset: Option<usize>,
    layout: Option<usize>,
) -> Array<Tensor<T>> {

    'checkp11'.print();
    let seq_length = *(*X).shape[0];
    // let rank = (*X).shape.len();
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
    
    let (mut p_i, mut p_o, mut p_f) = {
        let p_split = split_tensor(P, 3, 0);
        (*p_split[0], *p_split[1], *p_split[2])
    };





    let w_transposed = W.transpose(axes: reverse_axes(*W.shape));
    let r_transposed = R.transpose(axes: reverse_axes(*R.shape));

    'checkp17'.print();

    let mut H_t = H_0;
    let mut C_t = C_0;
    let mut H = H_0;

    'checkp18'.print();
    'the x swapped outside'.print();
    (*(*X).shape[0]).print();

    let X_segment = split_tensor(X, *(*X).shape[0], 0);
    'checkp19'.print();
    (X_segment).len().print();
    let mut z = 0;
    while z < (X_segment).len() {
        'checkp20'.print();
        


        'checkp21'.print();
        let (mut b_i, mut b_o) = {
            let b_split = split_tensor(B, 2, 0);
            (*b_split[0], *b_split[1])
        };

       

        let gates = (X_segment[z].unsqueeze(axes: array![0].span()).matmul(@w_transposed)
            + H_t.matmul(@r_transposed).unsqueeze(axes: array![0].span())
            + (b_i + b_o));

        let (mut i, mut o, mut f, mut c) = {
                let gates_split = split_tensor(@gates, 4, gates.shape.len() - 1);
                (*gates_split.at(0), *gates_split.at(1), *gates_split.at(2), *gates_split.at(3))
            };

        'checkp22'.print();
        let mut e1 = (i + p_i) + *C_t;
        i = f(e1);
        'checkp22aa'.print();
        let mut e2 = (f + p_f) + *C_t;
        f = f(e2);
        c = g(@c);
        'checkp22bb'.print();
        
        
        let mut C = f * *C_t + i * c;

        let mut e3 = o + p_o + C;
        o = f(e3);

        H = @(o * h(@C));


        h_list.append(*H);
        H_t = H;
        C_t = @C;
        z += 1;
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

        'the yyy shape'.print();
        (*Y.shape[0]).print();
        (*Y.shape[1]).print();
        (*Y.shape[2]).print();
        (*Y.shape[3]).print();


        Y = concatenated.reshape(
            array![(*Y.shape[0]).into(), (*Y.shape[1]).into(), 
                   (*Y.shape[2]).into(), (*Y.shape[3]).into()].span(),
            false
        );

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


fn h<
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