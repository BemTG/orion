use core::traits::TryInto;
use core::debug::PrintTrait;
use alexandria_data_structures::array_ext::{SpanTraitExt};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{Tensor, TensorTrait, U32Tensor};
use orion::numbers::fixed_point::{core::{FixedTrait}};
use array::{ArrayTrait, SpanTrait};

use core::option::OptionTrait;


use orion::operators::tensor::{
     FP16x16Tensor, U32TensorAdd, FP16x16TensorSub, FP16x16TensorAdd,
    FP16x16TensorDiv, FP16x16TensorMul
};


use orion::numbers::fixed_point::implementations::fp16x16::core::{
    FP16x16, FP16x16Impl, FP16x16Add, FP16x16AddEq, FP16x16Sub, FP16x16Mul, FP16x16MulEq,
    FP16x16TryIntoU128, FP16x16PartialEq, FP16x16PartialOrd, FP16x16SubEq, FP16x16Neg, FP16x16Div,
    FP16x16IntoFelt252, FP16x16Print, HALF
};

use orion::operators::matrix::{MutMatrixTrait, MutMatrix, MutMatrixImpl};


use orion::operators::sequence::SequenceTrait;
use orion::operators::sequence::implementations::sequence_fp8x23::FP8x23Sequence;
use orion::operators::sequence::implementations::sequence_fp8x23wide::FP8x23WSequence;
use orion::operators::sequence::implementations::sequence_fp16x16::FP16x16Sequence;
use orion::operators::sequence::implementations::sequence_fp16x16wide::FP16x16WSequence;
use orion::operators::sequence::implementations::sequence_i8::I8Sequence;
use orion::operators::sequence::implementations::sequence_i32::I32Sequence;
use orion::operators::sequence::implementations::sequence_u32::U32Sequence;
use orion::operators::sequence::implementations::sequence_bool::BoolSequence;



use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::U32TensorPartialEq;
// ---------------------------------------------------------------------------------------------
use orion::operators::tensor::I8Tensor;
// use orion::numbers::{IntegerTrait, i8};




fn splittosequence<
    T,
    +Copy<T>,
    +Drop<T>,
    +TensorTrait<T>,
    +PartialOrd<T>,
    +PartialEq<T>,
    +PartialEq<Tensor<T>>,
    +PartialOrd<Tensor<T>>
>(
    self: @Tensor<T>, split: Option<Tensor<usize>>, axis:usize, keepdims:usize ) -> Array<Tensor<T>> {

    
    let split_defined = match split {
        Option::Some(value) => {
            true
        },
        Option::None => false,
    };

    let mut split_length: Array<usize> = array![];

    let mut i: usize = 0;
    if split_defined ==false{
        loop {
            if (i>=*(*self).shape.at(axis)) {    
                break;
            }
            split_length.append(1);
            i += 1;
            'in here'.print();
        };

    }
    //scalar
    else if split.unwrap().shape.len()==0 {

        let mut dim = *(*self).shape.at(axis);
        let length = split.unwrap().data.at(0);
        let mut n = dim/*length;
        let mut i: usize = 0;
        loop {
            if i>=n {
                break;
            }
            split_length.append(*length);
            i += 1;
            'the len len'.print();
        };
        
        // split_length.len().print();
        

        let mut left = dim - *length * n;
        // left.print();

        if left >0 {
            split_length.append(left);
            'the jump'.print();
            split_length.len().print();
            (*split_length.at(0)).print();
            (*split_length.at(1)).print();
            // (*split_length.at(2)).print();
           
        }
    }

    else {
        let mut i: usize = 0;
        loop {
            // 'heyhey'.print();
            // split.unwrap().data.len().print();
            if i>=split.unwrap().data.len() {
                break;
            }
            split_length.append(*split.unwrap().data.at(i));
            i += 1;
            'ova here'.print();
        };
       
    }

    let mut final_array: Array<Tensor<T>> = array![];
    let mut splited_t: Array<Tensor<T>> = array![];
    let mut sli: MutMatrix<usize> = MutMatrixImpl::new((*self).shape.len(), 2);    
    let mut pos: usize = 0;
    let mut i = 0;

    loop {
        if (i>=(*self).shape.len()) {
            break;
        }
        let s: usize = *(*self).shape.at(i);
        sli.set(i,0,0); 
        sli.set(i,1,s); 
        i += 1;
        i.print();
    };

    let mut i: usize = 0;
     'jj'.print();
    //  (*split.unwrap().shape.at(0)).print();

     // let mut split = split.unwrap();
     'fifi'.print();
     // (*split.data.at(0)).print();

    //  if split.shape.len()== 0 {

    //     split  = TensorTrait::new(shape: array![1].span(), data: array![2].span());
    //  };

    
    loop {
        if (i>=split_length.len()) {
            break;
        }
        let spl: usize = *split_length.at(i);
       
        sli.set(axis, 0, pos);
        pos += spl; 
        sli.set(axis, 1, pos);
        

        let end_ele_0 = match sli.get(1, 0) {
                    Option::Some(res) => {
                        res
                    },
                    Option::None(_) => {
                        assert(false, 'Get end_ele_0 is failed');
                        0
                    },
        };
        let end_ele_1 = match sli.get(1, 1) {
                    Option::Some(res) => {
                        res
                    },
                    Option::None(_) => {
                        assert(false, 'Get end_ele_0 is failed');
                        0
                    },
        };
        let starts: Span<usize> = array![sli.get(0,0).unwrap(),end_ele_0].span();
        let ends: Span<usize> = array![ sli.get(0,1).unwrap(), end_ele_1].span();
        let axes: Option<Span<usize>> = Option::None(());
        let steps: Option<Span<usize>> = Option::None(());
        let mut sub_t: Tensor<T> = (self).slice(starts, ends, axes, steps); 
        'koko'.print();
        let mut len = sub_t.shape.len();
        // let mut final_result = SequenceTrait::sequence_construct(tensors: array![sub_t]);
        splited_t.append(sub_t);
        'opopo'.print();
        i += 1;

    };

    
    // let mut final_result = SequenceTrait::sequence_construct(splited_t);
    let mut final_result = splited_t;



    if keepdims ==0  && split_defined == false { 
        let mut splited_t2: Array<Tensor<T>> = array![];
        'jiji'.print();
        let mut i: usize = 0;
        loop {
        if (i>=(final_result).len()) {
            break;
        }
        // let mut splited_t: Array<Tensor<T>> = array![];
        let mut tmp = final_result.at(i);
        let mut tensor_with_squeeze = tmp.squeeze(axes: Option::None(()));
        splited_t2.append(tensor_with_squeeze);
        i+=1;
 
    };
     // final_result = SequenceTrait::sequence_construct(splited_t2);
     final_result = splited_t2;

    // let tensor1:Tensor<T>  = TensorTrait::new(shape: array![2, 2].span(), data: array![0, 1, 2, 3].span());
    // let tensor2:Tensor<T>  = TensorTrait::new(shape: array![2, 2].span(), data: array![4, 5, 6, 7].span());
    let result22: Array<Tensor<T>> = SequenceTrait::sequence_construct(splited_t2);

    
    };

    return final_result;
    // return reuslt22;


    }

