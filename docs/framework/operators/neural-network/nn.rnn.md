# NNTrait::rnn

```rust
    rnn(
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
    layout: Option<usize>
    linear_before_reset: Option<usize>,
    n_outputs: Option<usize>
    ) -> Array<Tensor<T>>
```

Computes a one-layer Recurrent Neural Network (RNN).

## Args

* `X`(`@Tensor<T>`) - The input sequences packed into one 3-D tensor with shape [seq_length, batch_size, input_size].
* `W`(`@Tensor<T>`) - The weight tensor for input gate with shape [num_directions, hidden_size, input_size].
* `R`(`@Tensor<T>`) - The recurrence weight tensor with shape [num_directions, hidden_size, hidden_size].
* `B`(`Option<Tensor<T>>`) - The bias tensor for input gate with shape [num_directions, 2*hidden_size]. Optional: If not specified - assumed to be 0.
* `sequence_length`(`Option<Tensor<T>>`) - Optional tensor specifying lengths of the sequences in a batch.
* `initial_h`(`Option<Tensor<T>>`) - Optional initial value of the hidden.
* `activation_alpha`(`Option<Array<Tensor<T>>>`) - Optional scaling values for activation functions.
* `activation_beta`(`Option<Array<Tensor<T>>>`) - Optional scaling values for activation functions.
* `activations`(`Option<ACTIVATIONS>`) - One (or two if bidirectional) activation function for the cell. Tanh is default.
* `clip`(`Option<T>`) - Cell clip threshold. Optional.
* `direction`(`Option<DIRECTION>`) - RNN direction (forward, reverse, or bidirectional).
* `hidden_size`(`Option<usize>`) - Number of neurons in the hidden layer.
* `layout`(`Option<usize>`) - The shape format of inputs and outputs (0 or 1).
* `linear_before_reset`(`Option<usize>`) - When computing the output of the hidden gate, apply the linear transformation before multiplying by the output of the reset gate.
* `n_outputs`(`Option<usize>`) - The number of outputs to return.

## Returns

An `Array<Tensor<T>>` containing:
* `Y`: A tensor that concats all the intermediate output values of the hidden.
* `Y_h`: The last output value of the hidden.


## Examples

```rust
use orion::operators::nn::NNTrait;
use orion::numbers::FixedTrait;
use orion::operators::nn::FP16x16NN;
use orion::numbers::FP16x16;
use orion::operators::tensor::{Tensor, TensorTrait, FP16x16Tensor};
use orion::operators::nn::functional::rnn::{ACTIVATIONS, DIRECTION};

fn rnn_example() -> Array<Tensor<FP16x16>> {
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
    let X = TensorTrait::new(shape.span(), data.span());

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 6553, sign: false });
    data.append(FP16x16 { mag: 6553, sign: false });
    data.append(FP16x16 { mag: 6553, sign: false });
    data.append(FP16x16 { mag: 6553, sign: false });
    let W = TensorTrait::new(shape.span(), data.span());

    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(2);
    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 6553, sign: false });
    data.append(FP16x16 { mag: 6553, sign: false });
    data.append(FP16x16 { mag: 6553, sign: false });
    data.append(FP16x16 { mag: 6553, sign: false });
    let R = TensorTrait::new(shape.span(), data.span());

    return NNTrait::rnn(
        @X, @R, @W,
        Option::None(()), Option::None(()), Option::None(()),
        Option::None(()), Option::None(()),
        Option::None(()),
        Option::None(()),
        Option::None(()),
        Option::None(()),
        Option::None(()),
        Option::None(()),
        Option::Some(2),
    );
}

>>> [
        [
            [
                [
                  [19091, 19091],
                  [39607, 39607],
                  [52461, 52461]
                ],
            ]
        ],

        [
            [
               [19091, 19091],
               [39607, 39607],
               [52461, 52461]
             ],
        ]
    ]
```
