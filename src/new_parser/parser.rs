use core::fmt::Debug;
use core::mem::ManuallyDrop;
use std::collections::BTreeMap;

use crate::new_parser::fallback::FallbackIndexer;
use crate::new_parser::simd::Avx2Indexer;

use crate::jentry::JEntry;

use crate::builder::ArrayBuilder;
use crate::builder::ObjectBuilder;
use crate::builder::Entry;


const EVEN_BITS_MASK: u64 = 0x5555_5555_5555_5555;
const ODD_BITS_MASK: u64 = !EVEN_BITS_MASK;

const SIMD_INPUT_LENGTH: usize = 64;

/// static cast to an u32
#[macro_export]
macro_rules! static_cast_u32 {
    ($v:expr) => {
        ::std::mem::transmute::<_, u32>($v)
    };
}

/// static cast to an i32
#[macro_export]
macro_rules! static_cast_i32 {
    ($v:expr) => {
        std::mem::transmute::<_, i32>($v)
    };
}

/// static cast to an u64
#[macro_export]
macro_rules! static_cast_u64 {
    ($v:expr) => {
        ::std::mem::transmute::<_, u64>($v)
    };
}

/// static cast to an i64
#[macro_export]
macro_rules! static_cast_i64 {
    ($v:expr) => {
        ::std::mem::transmute::<_, i64>($v)
    };
}

#[derive(Copy, Eq, PartialEq, Clone, Debug)]
pub struct ParseError;

impl core::fmt::Display for ParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("invalid json sequence")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ParseError {}

// pub trait HighwayHash: Sized {
pub trait InnerIndexer: Sized {
    fn cmp_mask(&mut self, data: &[u8]) -> Result<(u64, u64), ParseError>;

    fn compute_quote_mask(&self, _quote_bits: u64) -> u64;

    fn find_whitespace_and_structurals(&self, _whitespace: &mut u64, _structurals: &mut u64);

    fn index_extract(&self, structurals: u64, idx: u32, base: &mut Vec<u32>);
}

#[derive(Debug, Clone, PartialEq)]
pub enum State {
    Init,
    ObjectBegin,
    ObjectKey,
    ObjectField,
    ObjectValue,
    ObjectContinue,
    ObjectEnd,
    ArrayBegin,
    ArrayValue,
    ArrayContinue,
    ArrayEnd,
    Finish,
}

#[derive(Debug, Clone)]
pub enum Tape {
    Placeholder,
    Null,
    True,
    False,
    // Number(Number),
    // String(Vec<StringAtom>),
    Number(f64),
    KeyString(String),
    String((usize, usize)),
    Array(Vec<usize>),
    Object(BTreeMap<String, usize>),
}

pub struct IndexIterator<'a> {
    data: &'a [u8],
    index: usize,
    indices: &'a Vec<u32>,
}

impl<'a> IndexIterator<'a> {
    pub fn new(data: &'a [u8], indices: &'a Vec<u32>) -> Self {
        IndexIterator {
            data,
            index: 0,
            indices,
        }
    }
}

impl<'a> Iterator for IndexIterator<'a> {
    type Item = (usize, char);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.indices.len() {
            None
        } else {
            let idx = self.indices[self.index] as usize;
            self.index += 1;
            Some((idx, self.data[idx] as char))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.indices.len() - self.index, Some(self.indices.len()))
    }
}

union IndexerChoices {
    #[cfg(target_arch = "x86_64")]
    avx2: ManuallyDrop<Avx2Indexer>,
    //#[cfg(target_arch = "x86_64")]
    // sse: ManuallyDrop<SseHash>,
    fallback: ManuallyDrop<FallbackIndexer>,
}

pub struct Parser {
    tag: u8,
    //inner: IndexerChoices,
    inner: Avx2Indexer,
    structural_indexes: Vec<u32>,
}

impl Parser {
    pub fn new() -> Self {
        /**
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let avx2 = ManuallyDrop::new(Avx2Indexer::new());
                return Self {
                    tag: 1,
                    inner: IndexerChoices { avx2 },
                    structural_indexes: Vec::new(),
                };
            }
        }

        let fallback = ManuallyDrop::new(FallbackIndexer::new());
        Self {
            tag: 0,
            inner: IndexerChoices { fallback },
            structural_indexes: Vec::new(),
        }
        */

        let avx2 = Avx2Indexer::new();
        Self {
            tag: 1,
            inner: avx2,
            structural_indexes: Vec::new(),
        }
    }

    // stage 0
    // 检查是否是合法的 utf8

    // stage 1
    // 1. 获取到 backslash_bits 和 quote_bits  需要 SIMD
    // fn cmp_mask()

    // 2. 获取需要转义的字符串位置，不需要 SIMD
    // fn find_odd_backslash(
    // 3. 找到 空白字符和结构字符，需要 SIMD
    // fn classify(

    // 4. 处理索引结构，不需要 SIMD
    // fn identify_index

    // 5. 构造出 index，需要 SIMD
    // fn index_extract(

    // stage 2
    // 6. 遍历所有的 index，构造 Tape
    // fn build_tape(

    // 7. 遍历 Tape，构造出 jsonb

    pub fn parse(&mut self, data: &[u8]) -> Result<Vec<u8>, ParseError> {
        self.stage0(data)?;
        self.stage1(data)?;
/**
        let tapes = self.stage2(data)?;

        let mut buf = Vec::new();
        let entry = self.stage3(0, &tapes, data)?;

        match entry {
            Entry::ArrayBuilder(builder) => {
                builder.build_into(&mut buf);
            }
            Entry::ObjectBuilder(builder) => {
                builder.build_into(&mut buf);
            }
            _ => todo!(),
        }
*/
        //println!("buf={:?}", buf);
        let buf = Vec::new();

        Ok(buf)
    }

    fn stage3<'a>(&'a self, i: usize, tapes: &'a Vec<Tape>, data: &'a [u8]) -> Result<Entry, ParseError> {
        let tape = unsafe { tapes.get_unchecked(i) };
        let entry = match tape {
            Tape::Object(objs) => {
                let mut obj_builder = ObjectBuilder::new();
                for (key, idx) in objs {
                    let entry = self.stage3(*idx, &tapes, data)?;
                    match entry {
                        Entry::ArrayBuilder(builder) => {
                            obj_builder.push_array(key, builder);
                        }
                        Entry::ObjectBuilder(builder) => {
                            obj_builder.push_object(key, builder);
                        }
                        Entry::Raw(jentry, val) => {
                            obj_builder.push_raw(key, jentry, val);
                        }
                        _ => {}
                    }
                }
                Entry::ObjectBuilder(obj_builder)
            }
            Tape::Array(indices) => {
                let mut arr_builder = ArrayBuilder::new(indices.len());
                for idx in indices {
                    let entry = self.stage3(*idx, &tapes, data)?;
                    match entry {
                        Entry::ArrayBuilder(builder) => {
                            arr_builder.push_array(builder);
                        }
                        Entry::ObjectBuilder(builder) => {
                            arr_builder.push_object(builder);
                        }
                        Entry::Raw(jentry, val) => {
                            arr_builder.push_raw(jentry, val);
                        }
                        _ => {}
                    }
                }
                Entry::ArrayBuilder(arr_builder)
            }
            Tape::Null => {
                let entry = Entry::Raw(JEntry::make_null_jentry(), &[]);
                entry
            }
            Tape::True => {
                let entry = Entry::Raw(JEntry::make_true_jentry(), &[]);
                entry
            }
            Tape::False => {
                let entry = Entry::Raw(JEntry::make_false_jentry(), &[]);
                entry
            }
            Tape::KeyString(s) => {
                let entry = Entry::Raw(JEntry::make_string_jentry(s.len()), s.as_bytes());
                entry
            }
            Tape::String((i, j)) => {
                let entry = Entry::Raw(JEntry::make_string_jentry(j - i - 1), &data[i+1..*j]);
                entry
            }
            Tape::Number(_) => {
                let entry = Entry::Raw(JEntry::make_null_jentry(), &[]);
                entry
            }
            _ => {
                todo!()
            }
        };
        Ok(entry)
    }

    // check invalid UTF8
    fn stage0(&self, data: &[u8]) -> Result<(), ParseError> {
        // don't support parse larger than 4g JSON
        if data.len() > u32::MAX as usize {
            // todo return
            println!("too big");
        }
        if simdutf8::basic::from_utf8(data).is_err() {
            return Err(ParseError);
        }

        Ok(())
    }

    // stage1 build indexer
    fn stage1(&mut self, data: &[u8]) -> Result<(), ParseError> {
        // 基本功能已经实现
        // 需要考虑一下如何将前面的结构合并到后面

        self.structural_indexes.clear();

        let mut idx = 0;

        let mut whitespace: u64 = 0;
        let mut structurals: u64 = 0;

        // persistent state across loop
        // does the last iteration end with an odd-length sequence of backslashes?
        // either 0 or 1, but a 64-bit value
        let mut prev_iter_ends_odd_backslash: u64 = 0;
        // does the previous iteration end inside a double-quote pair?
        let mut prev_iter_inside_quote: u64 = 0;

        let mut prev_iter_ends_pseudo_pred: u64 = 1;

        let mut tmpbuf: [u8; SIMD_INPUT_LENGTH] = [0x20; SIMD_INPUT_LENGTH];

        while idx < data.len() {
            let val = if idx + SIMD_INPUT_LENGTH < data.len() {
                // let val = &data[idx..idx + SIMD_INPUT_LENGTH];
                // println!("val.len={:?}  val={:?}", val.len(), val);
                // val
                &data[idx..idx + SIMD_INPUT_LENGTH]
            } else {
                unsafe {
                    tmpbuf
                        .as_mut_ptr()
                        .copy_from(data.as_ptr().add(idx), data.len() - idx)
                };
                // println!("tmpbuf.len={:?}  tmpbuf={:?}", tmpbuf.len(), tmpbuf);
                &tmpbuf
            };

            // 1. 获取到 backslash_bits 和 quote_bits  需要 SIMD
            let (backslash_bits, mut quote_bits) = self.cmp_mask(val)?;

            // println!("backslash      ={:#066b}", backslash_bits);
            // println!("quote          ={:#066b}", quote_bits);

            // 找到 odd 起点，这样字符串中的反斜杠就不用再过滤了
            let odd_ends =
                self.find_odd_backslash(backslash_bits, &mut prev_iter_ends_odd_backslash)?;
            // println!("odd_ends       ={:#066b}", odd_ends);

            // 这里再包装到一个函数里面
            // quote_bits &= !odd_ends;

            let quote_mask =
                self.compute_quote_mask(odd_ends, &mut quote_bits, &mut prev_iter_inside_quote);
            // println!("quote_bits     ={:#066b}", quote_bits);
            // println!("quote_mask     ={:#066b}", quote_mask);
            // println!("prev__quote    ={:#066b}", prev_iter_inside_quote);

            // 3. 找到 空白字符和结构字符，需要 SIMD
            self.find_whitespace_and_structurals(&mut whitespace, &mut structurals);
            // println!("whitespace     ={:#066b}", whitespace);
            // println!("structurals    ={:#066b}", structurals);

            self.identify_index(
                quote_bits,
                quote_mask,
                whitespace,
                &mut structurals,
                &mut prev_iter_ends_pseudo_pred,
            );
            // println!("f-structurals  ={:#066b}", structurals);
            // println!("\n");

            // self.index_extract(structurals, &mut self.structural_indexes);
            self.index_extract(structurals, idx as u32);

            idx += SIMD_INPUT_LENGTH;
        }

        // println!("fina base.len()  == {:?}", self.structural_indexes.len());
        // println!("fina base  == {:?}", self.structural_indexes);

        Ok(())
    }

    fn stage2(&mut self, data: &[u8]) -> Result<Vec<Tape>, ParseError> {
        let mut tapes = Vec::with_capacity(self.structural_indexes.len());
        let mut stack = Vec::new();

        let mut state = State::Init;
        let mut index_iter = IndexIterator::new(data, &self.structural_indexes);
        while let Some((i, c)) = index_iter.next() {
            // let o_state = state.clone();
            match c {
                '{' => {
                    if !matches!(state, State::Init | State::ObjectField | State::ObjectContinue | State::ArrayBegin | State::ArrayValue | State::ArrayContinue) {
                        println!("obj begin invalid state={:?}", state);
                    }
                    state = State::ObjectBegin;
                    let tape = Tape::Object(BTreeMap::<String, usize>::new());
                    stack.push((tape, tapes.len()));
                    tapes.push(Tape::Placeholder);
                }
                '}' => {
                    if !matches!(
                        state,
                        State::ObjectBegin
                            | State::ObjectValue
                            | State::ArrayEnd
                            | State::ObjectEnd
                    ) {
                        println!("obj end invalid state={:?}", state);
                    }
                    let (tape, obj_idx) = stack.pop().unwrap();

                    let old_tape = unsafe { tapes.get_unchecked_mut(obj_idx) };
                    *old_tape = tape;

                    // println!("obj statck={:?}", stack);
                    state = if !stack.is_empty() {
                        let (this_tape, _) = unsafe { stack.get_unchecked(stack.len() - 1) };
                        match this_tape {
                            Tape::Object(_) => State::ObjectValue,
                            Tape::Array(_) => State::ArrayValue,
                            _ => unreachable!(),
                        }
                    } else {
                        State::ObjectEnd
                    };
                }
                '[' => {
                    if !matches!(state, State::Init | State::ObjectField | State::ObjectContinue | State::ArrayBegin | State::ArrayValue | State::ArrayContinue) {
                        println!("arr begin invalid state={:?}", state);
                    }
                    state = State::ArrayBegin;
                    let tape = Tape::Array(Vec::<usize>::new());
                    stack.push((tape, tapes.len()));
                    tapes.push(Tape::Placeholder);
                }
                ']' => {
                    if !matches!(
                        state,
                        State::ArrayBegin | State::ArrayValue | State::ArrayEnd | State::ObjectEnd
                    ) {
                        println!("arr end invalid state={:?}", state);
                    }
                    let (tape, arr_idx) = stack.pop().unwrap();
                    let old_tape = unsafe { tapes.get_unchecked_mut(arr_idx) };
                    *old_tape = tape;
                    // println!("arr statck={:?}", stack);
                    state = if !stack.is_empty() {
                        let (this_tape, _) = unsafe { stack.get_unchecked(stack.len() - 1) };
                        match this_tape {
                            Tape::Object(_) => State::ObjectValue,
                            Tape::Array(_) => State::ArrayValue,
                            _ => unreachable!(),
                        }
                    } else {
                        State::ArrayEnd
                    };
                }
                ',' => match state {
                    State::ObjectValue => {
                        state = State::ObjectContinue;
                    }
                    State::ArrayValue => {
                        state = State::ArrayContinue;
                    }
                    _ => {
                        println!("\n\n\n---hdot invalid, state={:?}", state);
                    }
                },
                ':' => match state {
                    State::ObjectKey => {
                        state = State::ObjectField;
                    }
                    _ => {
                        println!("\n\n\n-----maohao invalid state={:?}", state);
                    }
                },
                _ => {
                    let atom_tape = match c {
                        'n' => Tape::Null,
                        't' => Tape::True,
                        'f' => Tape::False,
                        '"' => {
                            let (j, next_c) = index_iter.next().unwrap();
                            // println!("\n\nj={:?} nc={:?}", j, next_c);
                            if next_c != '"' {
                                return Err(ParseError);
                            }
                            if matches!(state, State::ObjectBegin | State::ObjectContinue) {
                                let v = &data[i + 1..j];
                                Tape::KeyString(String::from_utf8_lossy(v).to_string())
                            } else {
                                Tape::String((i, j))
                            }
                        }
                        '-' | '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' => {
                            let s = &data[i..];
                            let (x, _) = fast_float::parse_partial::<f64, _>(s).unwrap();
                            Tape::Number(x)
                        }
                        _ => {
                            // println!("todo");
                            Tape::Null
                        }
                    };
                    // println!("atom value={:?}", atom_tape);

                    match state {
                        State::Init => {
                            state = State::Finish;
                        }
                        State::ObjectBegin | State::ObjectContinue => {
                            state = State::ObjectKey;

                            let last_idx = stack.len() - 1;
                            let (parent_tap, _) = unsafe { stack.get_unchecked_mut(last_idx) };
                            match parent_tap {
                                Tape::Object(obj_tap) => match atom_tape {
                                    Tape::KeyString(ref s) => {
                                        obj_tap.insert(s.clone(), tapes.len());
                                    }
                                    _ => todo!(),
                                },
                                _ => todo!(),
                            }
                        }
                        State::ObjectField => {
                            state = State::ObjectValue;
                        }
                        State::ArrayBegin | State::ArrayContinue => {
                            state = State::ArrayValue;

                            let last_idx = stack.len() - 1;
                            let (parent_tap, _) = unsafe { stack.get_unchecked_mut(last_idx) };
                            match parent_tap {
                                Tape::Array(arr_tap) => {
                                    arr_tap.push(tapes.len());
                                }
                                _ => todo!(),
                            }
                        }
                        _ => {
                            println!("\n\n\n---invalid tap state={:?}", state);
                        }
                    }
                    if state != State::ObjectKey {
                        tapes.push(atom_tape);
                    }
                }
            }

            // println!("c={:?} old_state={:?} state={:?}", c, o_state, state);
        }

        // println!("tapes={:?}", tapes);
        // println!("stack={:?}", stack);

        Ok(tapes)
    }

    // 1. 获取到 backslash_bits 和 quote_bits  需要 SIMD
    fn cmp_mask(&mut self, data: &[u8]) -> Result<(u64, u64), ParseError> {
        /**
        match self.tag {
            0 => unsafe { &mut self.inner.fallback }.cmp_mask(data),
            1 => unsafe { &mut self.inner.avx2 }.cmp_mask(data),
            _ => unreachable!(),
        }
        */
        self.inner.cmp_mask(data)
    }

    // 2. 获取需要转义的字符串位置，不需要 SIMD
    fn find_odd_backslash(
        &self,
        backslash_bits: u64,
        prev_iter_ends_odd_backslash: &mut u64,
    ) -> Result<u64, ParseError> {
        // println!(
        //    "\nprev_iter_ends_odd_backslash={:#066b}",
        //    prev_iter_ends_odd_backslash
        //);

        // 这段包装到一个函数里面
        // let s = b & !(b << 1);
        let backslash_start = backslash_bits & !(backslash_bits << 1);
        // println!("backslash_start={:#066b}", backslash_start);

        // flip lowest if we have an odd-length run at the end of the prior
        // iteration
        let even_start_mask: u64 = EVEN_BITS_MASK ^ *prev_iter_ends_odd_backslash;

        // let es = s & e;
        // let even_start = backslash_start & EVEN_BITS_MASK;
        let even_start = backslash_start & even_start_mask;
        // println!("even_start     ={:#066b}", even_start);

        // let ec = b + es;
        // let even_carries = backslash_bits + even_start;
        let even_carries: u64 = backslash_bits.wrapping_add(even_start); // ec = b + es;
        // println!("even_carries   ={:#066b}", even_carries);

        // let ece = ec & !b;
        let even_carries_escape = even_carries & !backslash_bits;
        // println!("even_carries_e ={:#066b}", even_carries_escape);

        // let od1 = ece & !e;
        let odd_ends1 = even_carries_escape & ODD_BITS_MASK;
        // println!("odd_ends1      ={:#066b}", odd_ends1);

        // let os = s & o;
        // let odd_start = backslash_start & ODD_BITS_MASK;
        let odd_start = backslash_start & !even_start_mask;
        // println!("odd_start      ={:#066b}", odd_start);

        // must record the carry-out of our odd-carries out of bit 63; this
        // indicates whether the sense of any edge going to the next iteration
        // should be flipped
        // let oc = b + os;
        // let odd_carries = backslash_bits + odd_start;
        let (mut odd_carries, iter_ends_odd_backslash) = backslash_bits.overflowing_add(odd_start); // oc = b + os;
        // println!("odd_carries    ={:#066b}", odd_carries);

        odd_carries |= *prev_iter_ends_odd_backslash;
        // push in bit zero as a potential end
        // if we had an odd-numbered run at the
        // end of the previous iteration
        *prev_iter_ends_odd_backslash = u64::from(iter_ends_odd_backslash);

        // let oce = oc & !b;
        let odd_carries_escape = odd_carries & !backslash_bits;
        // println!("odd_carries_e  ={:#066b}", odd_carries_escape);

        // let od2 = oce & e;
        let odd_ends2 = odd_carries_escape & EVEN_BITS_MASK;
        // println!("odd_ends2      ={:#066b}", odd_ends2);

        // let od1 | od2;
        let odd_ends = odd_ends1 | odd_ends2;
        // println!("odd_ends       ={:#066b}", odd_ends);

        Ok(odd_ends)
    }

    fn compute_quote_mask(
        &mut self,
        odd_ends: u64,
        quote_bits: &mut u64,
        prev_iter_inside_quote: &mut u64,
    ) -> u64 {
        *quote_bits &= !odd_ends;
/**
        let mut quote_mask = match self.tag {
            0 => unsafe { &mut self.inner.fallback }.compute_quote_mask(*quote_bits),
            1 => unsafe { &mut self.inner.avx2 }.compute_quote_mask(*quote_bits),
            _ => unreachable!(),
        };
*/
        let mut quote_mask = self.inner.compute_quote_mask(*quote_bits);

        quote_mask ^= *prev_iter_inside_quote;

        *prev_iter_inside_quote = unsafe { static_cast_u64!(static_cast_i64!(quote_mask) >> 63) };

        quote_mask
    }

    // 3. 找到 空白字符和结构字符，需要 SIMD
    fn find_whitespace_and_structurals(&mut self, whitespace: &mut u64, structurals: &mut u64) {
        /**
        match self.tag {
            0 => unsafe { &mut self.inner.fallback }
                .find_whitespace_and_structurals(whitespace, structurals),
            1 => unsafe { &mut self.inner.avx2 }
                .find_whitespace_and_structurals(whitespace, structurals),
            _ => unreachable!(),
        }
        */
        self.inner.find_whitespace_and_structurals(whitespace, structurals)
    }

    // 4. 处理索引结构，不需要 SIMD
    fn identify_index(
        &mut self,
        quote_bits: u64,
        quote_mask: u64,
        whitespace: u64,
        structurals: &mut u64,
        prev_iter_ends_pseudo_pred: &mut u64,
    ) {
        // 再包装一个函数，处理索引结构

        // mask off anything inside quotes
        *structurals &= !quote_mask;
        // add the real quote bits back into our bitmask as well, so we can
        // quickly traverse the strings we've spent all this trouble gathering
        *structurals |= quote_bits;

        let pseudo_pred: u64 = *structurals | whitespace;

        let shifted_pseudo_pred: u64 = (pseudo_pred << 1) | *prev_iter_ends_pseudo_pred;
        *prev_iter_ends_pseudo_pred = pseudo_pred >> 63;
        let pseudo_structurals: u64 = shifted_pseudo_pred & (!whitespace) & (!quote_mask);
        //*structurals |= pseudo_structurals;

        // let t1 = quote_bits & !quote_mask;
        // let t2 = !t1;

        // println!("structurals    ={:#066b}", *structurals);

        // 为什么去掉？留着不好吗？
        // now, we've used our close quotes all we need to. So let's switch them off
        // they will be off in the quote mask and on in quote bits.
        //*structurals &= !(quote_bits & !quote_mask);

        // println!("t1             ={:#066b}", t1);
        // println!("t2             ={:#066b}", t2);
        // println!("structurals    ={:#066b}", *structurals);
        *structurals |= pseudo_structurals;
    }

    // fn index_extract(&mut self, structurals: u64, base: &mut Vec<u32>) {
    fn index_extract(&mut self, structurals: u64, idx: u32) {
        /**
        match self.tag {
            0 => unsafe { &mut self.inner.fallback }.index_extract(
                structurals,
                idx,
                &mut self.structural_indexes,
            ),
            1 => unsafe { &mut self.inner.avx2 }.index_extract(
                structurals,
                idx,
                &mut self.structural_indexes,
            ),
            _ => unreachable!(),
        }
        */
        self.inner.index_extract(
            structurals,
            idx,
            &mut self.structural_indexes,
        )
    }
}

