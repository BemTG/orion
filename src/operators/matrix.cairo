use core::array::ArrayTrait;
use core::option::OptionTrait;

use orion::numbers::NumberTrait;
use orion::operators::vec::{VecTrait, NullableVec, NullableVecImpl};

struct MutMatrix<T> {
    data: NullableVec<T>,
    rows: usize,
    cols: usize,
}

impl MutMatrixDestruct<T, +Drop<T>> of Destruct<MutMatrix<T>> {
    fn destruct(self: MutMatrix<T>) nopanic {
        self.data.destruct()
    }
}

#[generate_trait]
impl MutMatrixImpl<
    T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>, +PartialOrd<T>
> of MutMatrixTrait<T> {
    /// Constructor for the Matrix
    fn new(rows: usize, cols: usize) -> MutMatrix<T> {
        MutMatrix { data: NullableVecImpl::new(), rows: rows, cols: cols }
    }

    /// Get the value at (row, col)
    fn get(ref self: MutMatrix<T>, row: usize, col: usize) -> Option<T> {
        if row >= self.rows || col >= self.cols {
            Option::None
        } else {
            self.data.get(row * self.cols + col)
        }
    }

    /// Set the value at (row, col)
    fn set(ref self: MutMatrix<T>, row: usize, col: usize, value: T) {
        if row < self.rows && col < self.cols {
            let index = row * self.cols + col;
            self.data.set(index, value)
        }
    }

    /// Returns the shape of the matrix as (rows, cols)
    fn shape(self: MutMatrix<T>) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Returns the index of the maximum value along the specified axis
    fn argmax(ref self: MutMatrix<T>, axis: usize) -> Span<usize> {
        assert(axis < 2, 'Invalid axis');

        let mut result: Array<usize> = ArrayTrait::new();
        if axis == 0 {
            let mut col: usize = 0;
            loop {
                if col == self.cols {
                    break;
                }

                let mut max_value = self.get(0, col);
                let mut max_value = match max_value {
                    Option::Some => { max_value.unwrap() },
                    Option::None => { NumberTrait::min_value() }
                };
                let mut max_index = 0;

                let mut row: usize = 1;
                loop {
                    if row == self.rows {
                        break;
                    }

                    let mut value = self.get(row, col);
                    let mut value = match value {
                        Option::Some => { value.unwrap() },
                        Option::None => { NumberTrait::min_value() }
                    };
                    if value > max_value {
                        max_value = value;
                        max_index = row;
                    }

                    row += 1;
                };

                result.append(max_index);

                col += 1;
            };

            return result.span();
        }

        let mut row: usize = 0;
        loop {
            if row == self.rows {
                break;
            }

            let mut max_value = self.get(row, 0);
            let mut max_value = match max_value {
                Option::Some => { max_value.unwrap() },
                Option::None => { NumberTrait::min_value() }
            };
            let mut max_index = 0;

            let mut col: usize = 1;
            loop {
                if col == self.cols {
                    break;
                }

                let mut value = self.get(row, col);
                let mut value = match value {
                    Option::Some => { value.unwrap() },
                    Option::None => { NumberTrait::min_value() }
                };
                if value > max_value {
                    max_value = value;
                    max_index = col;
                }

                col += 1;
            };

            result.append(max_index);

            row += 1;
        };

        return result.span();
    }

    /// Apply softmax to the matrix along the specified axis
    fn softmax<+AddEq<T>, +Div<T>>(ref self: MutMatrix<T>, axis: usize) -> MutMatrix<T> {
        assert(axis < 2, 'Invalid axis');

        let mut result = MutMatrixImpl::new(self.rows, self.cols);

        if axis == 0 {
            let mut col: usize = 0;
            loop {
                if col == self.cols {
                    break;
                }

                let mut sum_exp = NumberTrait::zero();
                let mut row: usize = 0;
                loop {
                    if row == self.rows {
                        break;
                    }

                    let value = self.get(row, col).unwrap().into();
                    sum_exp += value.exp();

                    row += 1;
                };

                row = 0;
                loop {
                    if row == self.rows {
                        break;
                    }

                    let value = self.get(row, col).unwrap().into();
                    let softmax_value = (value.exp() / sum_exp).into();
                    result.set(row, col, softmax_value);

                    row += 1;
                };

                col += 1;
            };
        } else {
            let mut row: usize = 0;
            loop {
                if row == self.rows {
                    break;
                }

                let mut sum_exp = NumberTrait::zero();
                let mut col: usize = 0;
                loop {
                    if col == self.cols {
                        break;
                    }

                    let value = self.get(row, col).unwrap().into();
                    sum_exp += value.exp();

                    col += 1;
                };

                col = 0;
                loop {
                    if col == self.cols {
                        break;
                    }

                    let value = self.get(row, col).unwrap().into();
                    let softmax_value = (value.exp() / sum_exp).into();
                    result.set(row, col, softmax_value);

                    col += 1;
                };

                row += 1;
            };
        }

        result
    }

    /// Apply softmax to the matrix along the specified axis, treating zeros as neutral
    fn softmax_zero<+AddEq<T>, +Div<T>, +PartialEq<T>>(
        ref self: MutMatrix<T>, axis: usize
    ) -> MutMatrix<T> {
        assert(axis < 2, 'Invalid axis');

        let mut result = MutMatrixImpl::new(self.rows, self.cols);

        if axis == 0 {
            let mut col: usize = 0;
            loop {
                if col == self.cols {
                    break;
                }

                let mut sum_exp = NumberTrait::zero();
                let mut row: usize = 0;
                loop {
                    if row == self.rows {
                        break;
                    }

                    let value = self.get(row, col).unwrap().into();
                    if value != NumberTrait::zero() {
                        sum_exp += value.exp();
                    }
                    row += 1;
                };

                row = 0;
                loop {
                    if row == self.rows {
                        break;
                    }

                    let value = self.get(row, col).unwrap().into();
                    if value != NumberTrait::zero() {
                        let softmax_value = (value.exp() / sum_exp).into();
                        result.set(row, col, softmax_value);
                    } else {
                        result.set(row, col, NumberTrait::zero());
                    }

                    row += 1;
                };

                col += 1;
            };
        } else {
            let mut row: usize = 0;
            loop {
                if row == self.rows {
                    break;
                }

                let mut sum_exp = NumberTrait::zero();
                let mut col: usize = 0;
                loop {
                    if col == self.cols {
                        break;
                    }

                    let value = self.get(row, col).unwrap().into();
                    if value != NumberTrait::zero() {
                        sum_exp += value.exp();
                    }
                    col += 1;
                };

                col = 0;
                loop {
                    if col == self.cols {
                        break;
                    }

                    let value = self.get(row, col).unwrap().into();

                    if value != NumberTrait::zero() {
                        let softmax_value = (value.exp() / sum_exp).into();
                        result.set(row, col, softmax_value);
                    } else {
                        result.set(row, col, NumberTrait::zero());
                    }

                    col += 1;
                };

                row += 1;
            };
        }

        result
    }

    /// Apply the sigmoid function to each element of the matrix
    fn sigmoid<+Mul<T>, +Add<T>, +Div<T>>(ref self: MutMatrix<T>) -> MutMatrix<T> {
        let mut result = MutMatrixImpl::new(self.rows, self.cols);

        let mut row: usize = 0;
        loop {
            if row == self.rows {
                break;
            }

            let mut col: usize = 0;
            loop {
                if col == self.cols {
                    break;
                }

                let value = self.get(row, col);
                if value.is_some() {
                    let value = NumberTrait::one()
                        / (NumberTrait::one() + (value.unwrap() * NumberTrait::neg_one()).exp());

                    result.set(row, col, value);
                }

                col += 1;
            };

            row += 1;
        };

        result
    }
}
