#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::new_parser::parser::InnerIndexer;
use crate::new_parser::parser::ParseError;
//use crate::new_parser::static_cast_i32;
//use crate::new_parser::static_cast_i64;
//use crate::new_parser::static_cast_u32;
use crate::static_cast_i32;
use crate::static_cast_i64;
use crate::static_cast_u32;

pub struct Avx2Indexer {
    backslash_mask: __m256i,
    quote_mask: __m256i,
    low_nibble_mask: __m256i,
    high_nibble_mask: __m256i,
    structural_shufti_mask: __m256i,
    whitespace_shufti_mask: __m256i,
    v0: __m256i,
    v1: __m256i,
}

impl Avx2Indexer {
    pub fn new() -> Self {
        let backslash_mask = unsafe { _mm256_set1_epi8('\\' as i8) };
        let quote_mask = unsafe { _mm256_set1_epi8('"' as i8) };

        let low_nibble_mask = unsafe {
            _mm256_setr_epi8(
                16, 0, 0, 0, 0, 0, 0, 0, 0, 8, 12, 1, 2, 9, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 8,
                12, 1, 2, 9, 0, 0,
            )
        };
        let high_nibble_mask = unsafe {
            _mm256_setr_epi8(
                8, 0, 18, 4, 0, 1, 0, 1, 0, 0, 0, 3, 2, 1, 0, 0, 8, 0, 18, 4, 0, 1, 0, 1, 0, 0, 0,
                3, 2, 1, 0, 0,
            )
        };
        let structural_shufti_mask = unsafe { _mm256_set1_epi8(0x7) };
        let whitespace_shufti_mask = unsafe { _mm256_set1_epi8(0x18) };

        let v0 = unsafe { _mm256_set1_epi8(0x0) };
        let v1 = unsafe { _mm256_set1_epi8(0x0) };

        Self {
            backslash_mask,
            quote_mask,
            low_nibble_mask,
            high_nibble_mask,
            structural_shufti_mask,
            whitespace_shufti_mask,
            v0,
            v1,
        }
    }
}

impl InnerIndexer for Avx2Indexer {
    fn cmp_mask(&mut self, data: &[u8]) -> Result<(u64, u64), ParseError> {
        let v0 = unsafe { _mm256_loadu_si256(data.as_ptr().cast::<__m256i>()) };
        let v1 = unsafe { _mm256_loadu_si256(data.as_ptr().add(32).cast::<__m256i>()) };

        self.v0 = v0;
        self.v1 = v1;

        let backslash = cmp(v0, v1, self.backslash_mask);
        let quote = cmp(v0, v1, self.quote_mask);

        Ok((backslash, quote))
    }

    fn compute_quote_mask(&self, quote_bits: u64) -> u64 {
        unsafe {
            _mm_cvtsi128_si64(_mm_clmulepi64_si128(
                _mm_set_epi64x(0, static_cast_i64!(quote_bits)),
                _mm_set1_epi8(-1_i8 /* 0xFF */),
                0,
            )) as u64
        }
    }

    fn find_whitespace_and_structurals(&self, whitespace: &mut u64, structurals: &mut u64) {
        unsafe {
            // do a 'shufti' to detect structural JSON characters
            // they are
            // * `{` 0x7b
            // * `}` 0x7d
            // * `:` 0x3a
            // * `[` 0x5b
            // * `]` 0x5d
            // * `,` 0x2c
            // these go into the first 3 buckets of the comparison (1/2/4)

            // we are also interested in the four whitespace characters:
            // * space 0x20
            // * linefeed 0x0a
            // * horizontal tab 0x09
            // * carriage return 0x0d
            // these go into the next 2 buckets of the comparison (8/16)

            let v_lo: __m256i = _mm256_and_si256(
                _mm256_shuffle_epi8(self.low_nibble_mask, self.v0),
                _mm256_shuffle_epi8(
                    self.high_nibble_mask,
                    _mm256_and_si256(_mm256_srli_epi32(self.v0, 4), _mm256_set1_epi8(0x7f)),
                ),
            );

            let v_hi: __m256i = _mm256_and_si256(
                _mm256_shuffle_epi8(self.low_nibble_mask, self.v1),
                _mm256_shuffle_epi8(
                    self.high_nibble_mask,
                    _mm256_and_si256(_mm256_srli_epi32(self.v1, 4), _mm256_set1_epi8(0x7f)),
                ),
            );
            let tmp_lo: __m256i = _mm256_cmpeq_epi8(
                _mm256_and_si256(v_lo, self.structural_shufti_mask),
                _mm256_set1_epi8(0),
            );
            let tmp_hi: __m256i = _mm256_cmpeq_epi8(
                _mm256_and_si256(v_hi, self.structural_shufti_mask),
                _mm256_set1_epi8(0),
            );

            let structural_res_0: u64 = u64::from(static_cast_u32!(_mm256_movemask_epi8(tmp_lo)));
            let structural_res_1: u64 = _mm256_movemask_epi8(tmp_hi) as u64;
            *structurals = !(structural_res_0 | (structural_res_1 << 32));

            let tmp_ws_lo: __m256i = _mm256_cmpeq_epi8(
                _mm256_and_si256(v_lo, self.whitespace_shufti_mask),
                _mm256_set1_epi8(0),
            );
            let tmp_ws_hi: __m256i = _mm256_cmpeq_epi8(
                _mm256_and_si256(v_hi, self.whitespace_shufti_mask),
                _mm256_set1_epi8(0),
            );

            let ws_res_0: u64 = u64::from(static_cast_u32!(_mm256_movemask_epi8(tmp_ws_lo)));
            let ws_res_1: u64 = _mm256_movemask_epi8(tmp_ws_hi) as u64;
            *whitespace = !(ws_res_0 | (ws_res_1 << 32));
        }
    }

    fn index_extract(&self, structurals: u64, idx: u32, base: &mut Vec<u32>) {
        // 再来一个函数，构造最后的 索引

        // 5. 构造出 index，需要 SIMD
        // let idx = 0;
        // let idx_minus_64 = idx.wrapping_sub(64);
        // let idx_minus_64 = idx as u32;
        let idx_minus_64 = idx;

        let mut l = base.len();

        let idx_64_v = unsafe {
            _mm256_set_epi32(
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
                static_cast_i32!(idx_minus_64),
            )
        };

        let mut bits = structurals;

        base.reserve(64);

        let cnt: usize = bits.count_ones() as usize;
        let final_len = l + cnt;

        // println!("v0             ={:#034b}", v0);
        // println!("v0             ={}", v0);
        // println!("v1             ={:#034b}", v1);
        // println!("v1             ={}", v1);
        // println!("v2             ={:#034b}", v2);
        // println!("v2             ={}", v2);
        // println!("v3             ={:#034b}", v3);
        // println!("v3             ={}", v3);
        // println!("v4             ={:#034b}", v4);
        // println!("v4             ={}", v4);
        // println!("v5             ={:#034b}", v5);
        // println!("v5             ={}", v5);
        // println!("v6             ={:#034b}", v6);
        // println!("v6             ={}", v6);
        // println!("v7             ={:#034b}", v7);
        // println!("v7             ={}", v7);
        while bits != 0 {
            let v0 = bits.trailing_zeros() as i32;
            bits &= bits.wrapping_sub(1);
            let v1 = bits.trailing_zeros() as i32;
            bits &= bits.wrapping_sub(1);
            let v2 = bits.trailing_zeros() as i32;
            bits &= bits.wrapping_sub(1);
            let v3 = bits.trailing_zeros() as i32;
            bits &= bits.wrapping_sub(1);
            let v4 = bits.trailing_zeros() as i32;
            bits &= bits.wrapping_sub(1);
            let v5 = bits.trailing_zeros() as i32;
            bits &= bits.wrapping_sub(1);
            let v6 = bits.trailing_zeros() as i32;
            bits &= bits.wrapping_sub(1);
            let v7 = bits.trailing_zeros() as i32;
            bits &= bits.wrapping_sub(1);
            let v: __m256i = unsafe { _mm256_set_epi32(v7, v6, v5, v4, v3, v2, v1, v0) };
            let v: __m256i = unsafe { _mm256_add_epi32(idx_64_v, v) };
            unsafe {
                _mm256_storeu_si256(
                    base.as_mut_ptr()
                        .add(l)
                        .cast::<std::arch::x86_64::__m256i>(),
                    v,
                )
            };

            // println!("base----={:?}", base);

            l += 8;
        }

        // println!("base={:?}", base);
        // println!("final_len={:?}", final_len);
        // We have written all the data
        unsafe { base.set_len(final_len) };
    }
}

fn cmp(v0: __m256i, v1: __m256i, mask: __m256i) -> u64 {
    unsafe {
        let cmp_0 = _mm256_cmpeq_epi8(v0, mask);
        let res_0 = u64::from(static_cast_u32!(_mm256_movemask_epi8(cmp_0)));
        let cmp_1 = _mm256_cmpeq_epi8(v1, mask);
        let res_1 = u64::from(static_cast_u32!(_mm256_movemask_epi8(cmp_1)));
        res_0 | (res_1 << 32)
    }
}
