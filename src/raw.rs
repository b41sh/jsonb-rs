use crate::error::*;

pub struct RawJsonb<B: AsRef<[u8]>>(pub B);

impl<B: AsRef<[u8]>> RawJsonb<B> {
	pub fn new(data: B) -> RawJsonb<B> {
		Self(data)
	}

	pub fn test(&self) -> Result<u8, Error> {
		Ok(0)
	}
}


