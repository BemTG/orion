
# NNTrait::gru

```rust
    gru(
        X: @Tensor<T>,
        W: @Tensor<T>,
        R: @Tensor<T>,
        B: Option<Tensor<T>>,
        sequence_length: Option<Tensor<T>>,
        initial_h: Option<Tensor<T>>,
        activation_alpha: Option<Array<Tensor<T>>>,
        activation_beta: Option<Array<Tensor<T>>>,
        activations: Option<orion::operators::nn::functional::gru::ACTIVATIONS>,
        clip: Option<T>,
        direction: Option<orion::operators::nn::functional::gru::DIRECTION>,
        hidden_size: Option<usize>,
        layout: Option<usize>,
        linear_before_reset: Option<usize>,
        n_outputs: Option<usize>
    ) -> Array<Tensor<T>>
```

Computes an one-layer GRU.

## Args

* `X`(`@Tensor<T>`) - The input sequences packed (and potentially padded) into one 3-D tensor with the shape of [seq_length, batch_size, input_size].
* `W`(`@Tensor<T>`) - The weight tensor for the gates. Concatenation of W[zrh] and WB[zrh] (if bidirectional) along dimension 0. This tensor has shape [num_directions, 3*hidden_size, input_size].
* `R`(`@Tensor<T>`) - The recurrence weight tensor. Concatenation of R[zrh] and RB[zrh] (if bidirectional) along dimension 0. This tensor has shape [num_directions, 3*hidden_size, hidden_size].
* `B`(`Option<Tensor<T>>`) - The bias tensor for the gates. Concatenation of [Wb[zrh], Rb[zrh]] and [WBb[zrh], RBb[zrh]] (if bidirectional) along dimension 0. This tensor has shape [num_directions, 6*hidden_size]. Optional: If not specified - assumed to be 0.
* `sequence_length`(`Option<Tensor<T>>`) - Optional tensor specifying lengths of the sequences in a batch. If unspecified - assumed all sequences in the batch to have length `seq_length`.
* `initial_h`(`Option<Tensor<T>>`) - Optional initial value of the hidden. If not specified - assumed to be 0.
* `activation_alpha`(`Option<Array<Tensor<T>>>`) - Optional scaling values used by the activation functions.
* `activation_beta`(`Option<Array<Tensor<T>>>`) - Optional scaling values used by the activation functions.
* `activations`(`Option<orion::operators::nn::functional::gru::ACTIVATIONS>`) - A list of 2 (or 4 if bidirectional) activation functions for update, reset, and hidden gates.
* `clip`(`Option<T>`) - Cell clip threshold. Clipping bounds the elements of a tensor in the range of [-threshold, +threshold] and is applied to the input of activations. No clip if not specified.
* `direction`(`Option<orion::operators::nn::functional::gru::DIRECTION>`) - Specify if the RNN is forward, reverse, or bidirectional.
* `hidden_size`(`Option<usize>`) - Number of neurons in the hidden layer.
* `layout`(`Option<usize>`) - The shape format of the input and output tensors. Must be 0 or 1.
* `linear_before_reset`(`Option<usize>`) - When computing the output of the hidden gate, apply the linear transformation before multiplying by the output of the reset gate.
* `n_outputs`(`Option<usize>`) - The number of outputs to return. 

## Returns

An `Array<Tensor<T>>` containing:
* `Y`: A tensor that concats all the intermediate output values of the hidden. It has shape `[seq_length, num_directions, batch_size, hidden_size]`.
* `Y_h`: The last output value of the hidden. It has shape `[num_directions, batch_size, hidden_size]`.

## Examples

```rust
use orion::operators::nn::NNTrait;
use orion::numbers::FixedTrait;
use orion::operators::nn::FP16x16NN;
use orion::numbers::FP16x16;
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor};
use orion::operators::nn::functional::gru::{ACTIVATIONS, DIRECTION};

fn gru_example() -> Array<Tensor<FP16x16>> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 131072, sign: false });
    data.append(FP16x16 { mag: 196608, sign: false });
    data.append(FP16x16 { mag: 262144, sign: false });
    data.append(FP16x16 { mag: 327680, sign: false });
    data.append(FP16x16 { mag: 393216, sign: false });
    let X = TensorTrait::new(shape.span(), data.span())

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 6553, sign: false });
    data.append(FP16x16 { mag: 6553, sign: false });
    data.append(FP16x16 { mag: 6553, sign: false });
    data.append(FP16x16 { mag: 6553, sign: false });
    data.append(FP16x16 { mag: 6553, sign: false });
    data.append(FP16x16 { mag: 6553, sign: false });
    let W = TensorTrait::new(shape.span(), data.span());
    /// 
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 6553, sign: false });
    data.append(FP16x16 { mag: 6553, sign: false });
    data.append(FP16x16 { mag: 6553, sign: false });
    let R = TensorTrait::new(shape.span(), data.span());

    return NNTrait::gru(
        @X, @W, @R,
        Option::None(()), Option::None(()), Option::None(()),
        Option::None(()), Option::None(()),
        Option::None(()), Option::None(()),
        Option::None(()), Option::None(()),
        Option::None(()), Option::None(()), 
        Option::Some(2)
    );
}

>>> [ 
       [
         [
          [[8124],
           [13142],
           [13101]]
         ]
       ],
       [
          [[8124],
           [13142],
           [13101]]
       ],
    ]
```
