use core::traits::TryInto;
use option::OptionTrait;
use array::ArrayTrait;
use array::SpanTrait;

fn u32_max(a: u32, b: u32) -> u32 {
    if a > b {
        a
    } else {
        b
    }
}

fn saturate<T, impl TCopy: Copy<T>, impl TDrop: Drop<T>, impl PartialOrdT: PartialOrd<T>>(
    min: T, max: T, x: T
) -> T {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}

fn assert_eq<T, impl TPartialEq: PartialEq<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>>(
    lhs: T, rhs: T
) {
    assert(lhs == rhs, 'should be equal');
}

fn assert_seq_eq<T, impl TPartialEq: PartialEq<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>>(
    lhs: Array<T>, rhs: Array<T>
) {
    assert(lhs.len() == rhs.len(), 'should be equal');

    let mut i = 0;
    loop {
        if i >= lhs.len() {
            break;
        }
        assert_eq(lhs[i], rhs[i]);
        i += 1;
    }
}
