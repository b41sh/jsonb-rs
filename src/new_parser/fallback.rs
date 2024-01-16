use crate::new_parser::parser::InnerIndexer;
use crate::new_parser::parser::ParseError;

pub struct FallbackIndexer {}

impl FallbackIndexer {
    pub fn new() -> Self {
        println!("new fallback");
        Self {}
    }
}

impl InnerIndexer for FallbackIndexer {
    fn cmp_mask(&mut self, _data: &[u8]) -> Result<(u64, u64), ParseError> {
        todo!()
    }

    fn compute_quote_mask(&self, _quote_bits: u64) -> u64 {
        todo!()
    }

    fn find_whitespace_and_structurals(&self, _whitespace: &mut u64, _structurals: &mut u64) {
        todo!()
    }

    fn index_extract(&self, _structurals: u64, _idx: u32, _base: &mut Vec<u32>) {
        todo!()
    }
}
