// Copyright 2023 Datafuse Labs.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::error::Error;
use std::str::FromStr;
use crate::parse_value;
use std::marker::PhantomData;


pub struct RawJsonb<B: AsRef<[u8]>>(pub B);

impl<B: AsRef<[u8]>> RawJsonb<B> {
    pub fn new(data: B) -> Self {
        Self(data)
    }
}


/**
impl TryFrom<RawJsonBuf> for RawJsonb {
    type Error = Error;

    fn try_from(raw: RawJsonBuf) -> Result<RawJsonb> {
        RawJsonb(raw.as_ref())
    }
}

impl TryFrom<&RawJsonb> for RawJsonBuf {
    type Error = Error;

    fn try_from(doc: &RawJsonb) -> Result<RawJsonBuf> {
        RawJsonBuf::new(doc)
    }
}
*/

//impl<B: AsRef<[u8]>> AsRef<RawJsonb<B>> for RawJsonBuf {
//    fn as_ref(&self) -> &RawJsonb<B> {
//        //RawJsonb(self.data.as_ref())
//        RawJsonb(unsafe { &*(self.data.as_ref() as *const [u8] as *const RawJsonb<B>) })
//    }
//}


pub struct RawJsonbBuf<B: AsRef<[u8]>> {
    data: Vec<u8>,
    _phantom: PhantomData<B>,
}

impl<B: AsRef<[u8]>> FromStr for RawJsonbBuf<B> {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let value = parse_value(s.as_bytes())?;
        let mut data = Vec::new();
        value.write_to_vec(&mut data);
        Ok(Self {
            data,
            _phantom: PhantomData
        })
    }
}

/**
impl<B: AsRef<[u8]>> AsRef<RawJsonb<B>> for RawJsonbBuf {
    fn as_ref(&self) -> &RawJsonb<B> {
        RawJsonb::new(&self.data)
    }
}
*/

//impl<B: AsRef<[u8]> + std::convert::From<std::vec::Vec<u8>>> RawJsonbBuf<B> {
//impl<B: AsRef<[u8]>> RawJsonbBuf<B> {
//impl<B: AsRef<[u8]> + std::convert::From<&[u8]>> RawJsonbBuf<B> {
impl<B: AsRef<[u8]> + for<'a> std::convert::From<&'a [u8]>> RawJsonbBuf<B> {
    fn to_raw_jsonb(&self) -> RawJsonb<B> {
        RawJsonb(self.data.as_slice().into())
    }
}



