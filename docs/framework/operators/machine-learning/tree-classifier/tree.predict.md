# TreeClassifierTrait::predict

```rust 
   fn predict(ref self: TreeClassifier<T>, features: Span<T>) -> T;
```

Predicts the target value for a set of features using the provided decision tree.

## Args

* `self`: A reference to the decision tree used for making the prediction.
* `features`: A span representing the features for which the prediction is to be made.

## Returns

The predicted target value.

## Type Constraints

Constrain input and output types to fixed point.

## Examples

```rust
use orion::operators::ml::{FP16x16TreeClassifier, TreeClassifierTrait, TreeClassifier};
use orion::numbers::{FP16x16, FixedTrait};

fn tree_classifier_example(tree: TreeClassifier<FP16x16>) {

 tree.predict(
       array![FixedTrait::new_unscaled(1, false), FixedTrait::new_unscaled(2, false),].span()
   );
   
}
```