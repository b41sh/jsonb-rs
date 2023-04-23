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

use std::borrow::Cow;

use super::constants::*;
use super::error::Error;
use super::error::ParseErrorCode;
use super::number::Number;
use super::util::parse_escaped_string;
use super::value::Object;
use super::value::Value;

use nom::error::ParseError;

use nom::{
    branch::alt,
    bytes::complete::{escaped, tag, take, take_while_m_n},
    character::complete::{alphanumeric1, char, i64, multispace0, one_of, u64},
    character::{is_alphabetic, is_hex_digit},
    combinator::{map, map_opt, map_res, opt, recognize},
    multi::{many0, many1, separated_list0},
    number::complete::hex_u32,
    sequence::{delimited, preceded, separated_pair, terminated, tuple},
    IResult,
};

use aho_corasick::{AhoCorasick, MatchKind};
use once_cell::sync::Lazy;

static PATTERNS: &[&str] = &["\\", "\""];

static TOKEN_START: Lazy<AhoCorasick> = Lazy::new(|| {
    AhoCorasick::builder()
        .match_kind(MatchKind::LeftmostFirst)
        .build(PATTERNS)
        .unwrap()
});

/// Parsing the input string to JSONB Value.
pub fn new_parse_value(input: &[u8]) -> Result<Value<'_>, Error> {
    match jsonb_value(input) {
        Ok((rest, value)) => {
            if !rest.is_empty() {
                return Err(Error::InvalidJsonPath);
            }
            Ok(value)
        }
        Err(nom::Err::Error(err) | nom::Err::Failure(err)) => {
            println!("------err={:?}", err.code);
            let ss = String::from_utf8(err.input.to_vec());
            println!("--rest={:?}", ss);
            Err(Error::InvalidJsonb)
        }
        Err(nom::Err::Incomplete(_)) => unreachable!(),
    }
}

fn jsonb_value(input: &[u8]) -> IResult<&[u8], Value<'_>> {
    map(delimited(multispace0, value, multispace0), |value| value)(input)
}

fn value(input: &[u8]) -> IResult<&[u8], Value<'_>> {
    /**
    alt((
        map(parse_null, |v| v),
        map(parse_true, |v| v),
        map(parse_false, |v| v),
        //map(u64, |v| Value::Number(Number::UInt64(v))),
        //map(i64, |v| Value::Number(Number::Int64(v))),
        //map(double, |v| Value::Number(Number::Float64(v))),
        map(number, |v| Value::Number(v)),
        map(string, |v| {
            //Value::String(Cow::Borrowed(unsafe { std::str::from_utf8_unchecked(v) }))
            //Value::String(v)
            v
        }),
        map(array_values, Value::Array),
        map(object_values, |kvs| {
            let mut obj = Object::new();
            for (k, v) in kvs {
                if let Value::String(k) = k {
                    let k = String::from(k);
                    //let k = String::from_utf8(k.to_vec()).unwrap();
                    obj.insert(k, v);
                }
            }
            Value::Object(obj)
        }),
    ))(input)
    */

    match input[0] {
        b'n' => parse_null(input),
        b't' => parse_true(input),
        b'f' => parse_false(input),
        b'0'..=b'9' | b'-' => parse_number(input),
        b'"' => string(input),
        b'[' => parse_array(input),
        b'{' => parse_object(input),
        _ => {
            //println!("--input[0]={:?}", input[0]);
            parse_null(input)
        },
    }
}

fn parse_number(input: &[u8]) -> IResult<&[u8], Value<'_>> {
    map(number, |v| Value::Number(v))(input)
}

fn parse_array(input: &[u8]) -> IResult<&[u8], Value<'_>> {
    map(array_values, |v| Value::Array(v))(input)
}

fn parse_object(input: &[u8]) -> IResult<&[u8], Value<'_>> {
    map(object_values, |kvs| {
        let mut obj = Object::new();
        for (k, v) in kvs {
            if let Value::String(k) = k {
                let k = String::from(k);
                //let k = String::from_utf8(k.to_vec()).unwrap();
                obj.insert(k, v);
            }
        }
        Value::Object(obj)
    })(input)
}


fn parse_null(input: &[u8]) -> IResult<&[u8], Value<'_>> {
    map(tag("null"), |_| Value::Null)(input)
}

fn parse_true(input: &[u8]) -> IResult<&[u8], Value<'_>> {
    map(tag("true"), |_| Value::Bool(true))(input)
}

fn parse_false(input: &[u8]) -> IResult<&[u8], Value<'_>> {
    map(tag("false"), |_| Value::Bool(false))(input)
}


/**

opt(char('-')) 0                    . many(digit) e/E opt(+/-)
opt(char('-')) oen(1-9) many(digit) . many(digit)
0
1-9

digit






number
    integer fraction exponent
integer
    digit
    onenine digits
    <code>'-'</code> digit
    <code>'-'</code> onenine digits

digits
    digit
    digit digits

digit
    <code>'0'</code>
    onenine

onenine
    <code>'1'</code> <code>.</code> <code>'9'</code>

fraction
    <code>""</code>
    <code>'.'</code> digits

exponent
    <code>""</code>
    <code>'E'</code> sign digits
    <code>'e'</code> sign digits

sign
    <code>""</code>
    <code>'+'</code>
    <code>'-'</code>


CREATE TABLE cars
(
    car_make varchar,
    car_model varchar[],
    car_type varchar[]
);

SELECT car_make, array_FILTER(car_model, car_type, x,y -> y = 'sedan') as sedans
FROM cars;


token value:sym<number> {
    '-'?
    [ 0 | <[1..9]> <[0..9]>* ]
    [ \. <[0..9]>+ ]?
    [ <[eE]> [\+|\-]? <[0..9]>+ ]?
}
*/

fn float(input: &[u8]) -> IResult<&[u8], &[u8]> {
    alt((
        // Case one: .42
        recognize(tuple((opt(char('-')), integer, fraction, exponent))),
        recognize(tuple((opt(char('-')), integer, fraction))),
        recognize(tuple((opt(char('-')), integer, exponent))),
    ))(input)
}

fn integer(input: &[u8]) -> IResult<&[u8], &[u8]> {
    alt((
        recognize(char('0')),
        recognize(tuple((one_of("123456789"), many0(one_of("0123456789"))))),
    ))(input)
}

fn neg_integer(input: &[u8]) -> IResult<&[u8], &[u8]> {
    recognize(tuple((char('-'), integer)))(input)
}

fn fraction(input: &[u8]) -> IResult<&[u8], &[u8]> {
    recognize(tuple((char('.'), many1(one_of("0123456789")))))(input)
}

fn exponent(input: &[u8]) -> IResult<&[u8], &[u8]> {
    recognize(tuple((
        one_of("eE"),
        opt(one_of("+-")),
        many1(one_of("0123456789")),
    )))(input)
}

#[derive(PartialEq)]
enum NumberState {
    Init,
    IsNegative,
    IsZero,
    IsInteger,
    IsFraction,
    IsExponent,
    IsExponentNumber,
}

fn number(input: &[u8]) -> IResult<&[u8], Number> {
/**
    alt((
        map(float, |s| {
            let v = unsafe { std::str::from_utf8_unchecked(s) };
            Number::Float64(fast_float::parse(v).unwrap())
        }),
        map(integer, |s| {
            let v = unsafe { std::str::from_utf8_unchecked(s) };
            match v.parse::<u64>() {
                Ok(n) => Number::UInt64(n),
                Err(_) => Number::Float64(fast_float::parse(v).unwrap()),
            }
        }),
        map(neg_integer, |s| {
            let v = unsafe { std::str::from_utf8_unchecked(s) };
            match v.parse::<i64>() {
                Ok(n) => Number::Int64(n),
                Err(_) => Number::Float64(fast_float::parse(v).unwrap()),
            }
        }),
    ))(input)
*/

    let mut i = 0;
    let mut is_negative = false;

    let mut state = NumberState::Init;
    while i < input.len() {
        match input[i] {
            b'-' => {
                if state == NumberState::Init {
                    is_negative = true;
                    state = NumberState::IsNegative;
                } else if state == NumberState::IsExponent {
                    state = NumberState::IsExponentNumber;
                } else {
                    todo!();
                }
            },
            b'+' => {
                if state == NumberState::IsExponent {
                    state = NumberState::IsExponentNumber;
                } else {
                    todo!();
                }
            }
            b'0' => {
                if state == NumberState::Init || state == NumberState::IsNegative {
                    state = NumberState::IsZero;
                }
            },
            b'1'..=b'9' => {
                if state == NumberState::Init || state == NumberState::IsNegative {
                    state = NumberState::IsInteger;
                } else if state == NumberState::IsZero {
                    todo!();
                }
            },
            b'.' => {
                if state == NumberState::IsZero || state == NumberState::IsInteger {
                    state = NumberState::IsFraction;
                } else {
                    todo!();
                }
            },
            b'e' | b'E' => {
                if state == NumberState::IsZero || state == NumberState::IsInteger || state == NumberState::IsFraction {
                    state = NumberState::IsExponent;
                } else {
                    todo!();
                }
            }
            _ => {
                break;
            }
        }

        i += 1;
    }
    let v = unsafe { std::str::from_utf8_unchecked(&input[..i]) };
    let rest = &input[i..];
    //println!("-----input={:?}", input);
    //println!("-----rest={:?}", rest);
    match state {
        NumberState::IsFraction | NumberState::IsExponent | NumberState::IsExponentNumber => {
            //println!("float v={:?}", v);
            let n = Number::Float64(fast_float::parse(v).unwrap());
            //println!("float n={:?}", n);
            Ok((rest, n))
        }
        NumberState::IsNegative | NumberState::IsZero | NumberState::IsInteger => {
            //println!("integer v={:?}", v);
            if is_negative {
                let n = match v.parse::<i64>() {
                    Ok(n) => Number::Int64(n),
                    Err(_) => Number::Float64(fast_float::parse(v).unwrap()),
                };
                //println!("integer n={:?}", n);
                Ok((rest, n))
            } else {
                let n = match v.parse::<u64>() {
                    Ok(n) => Number::UInt64(n),
                    Err(_) => Number::Float64(fast_float::parse(v).unwrap()),
                };
                //println!("uinteger n={:?}", n);
                Ok((rest, n))
            }
        }
        _ => todo!(),
    }
}








fn raw_string(input: &[u8]) -> IResult<&[u8], &[u8]> {
    escaped(alphanumeric1, '\\', one_of("\"n\\"))(input)
}

fn stringg(input: &[u8]) -> IResult<&[u8], &[u8]> {
    // TODO: support special characters and unicode characters.
    delimited(char('"'), raw_string, char('"'))(input)
}

//92, 117, 48, 48, 100, 49,
//\   u    1   2   3    4
//92, 117, 48, 48, 56, 55,

fn escaped_char(input: &[u8]) -> IResult<&[u8], char> {
    alt((
        map(preceded(char('u'), hex_u32), |hex| {
            char::from_u32(hex).unwrap()
        }),
        map(char('"'), |_| QU),
        map(char('\\'), |_| BS),
        map(char('/'), |_| SD),
        map(char('b'), |_| BB),
        map(char('f'), |_| FF),
        map(char('n'), |_| NN),
        map(char('r'), |_| RR),
        map(char('t'), |_| TT),
    ))(input)
}

//fn string(input: &[u8]) -> IResult<&[u8], Value<'_>> {
//fn string<'a>(input: &'a [u8]) -> IResult<&'a [u8], Cow<'a>> {
//fn string(input: &[u8]) -> IResult<&[u8], Cow<'_>> {
fn string(input: &[u8]) -> IResult<&[u8], Value<'_>> {
    if input.is_empty() || input[0] != b'"' {
        return Err(nom::Err::Error(nom::error::Error::from_error_kind(
            input,
            nom::error::ErrorKind::SeparatedList,
        )));
    }
    //let ss = String::from_utf8(input.to_vec());
    //println!("input={:?}", ss);

    let start = 1;
    let mut offset = 1;

    if let Some(mat) = TOKEN_START.find(&input[offset..]) {
        let end = mat.start();
        if mat.pattern() == 0.into() {
            let mut str_buf =
                unsafe { String::from_utf8_unchecked(input[start..end + offset].to_vec()) };
            let (mut rest, c) = escaped_char(&input[end + 2..]).unwrap();
            str_buf.push(c);
            while let Some(mat) = TOKEN_START.find(rest) {
                let end = mat.start();
                let s = unsafe { std::str::from_utf8_unchecked(&rest[..end]) };
                str_buf.extend(s.chars());
                if mat.pattern() == 0.into() {
                    let (restt, c) = escaped_char(&rest[end + 1..]).unwrap();
                    str_buf.push(c);
                    rest = restt;
                    continue;
                }
                let rest = &rest[end + 1..];
                return Ok((rest, Value::String(Cow::Owned(str_buf))));
            }
            return Err(nom::Err::Error(nom::error::Error::from_error_kind(
                input,
                nom::error::ErrorKind::SeparatedList,
            )));
        }
        let s = unsafe { std::str::from_utf8_unchecked(&input[start..end + offset]) };
        let rest = &input[end + offset + 1..];
        return Ok((rest, Value::String(Cow::Borrowed(s))));
    }

    return Err(nom::Err::Error(nom::error::Error::from_error_kind(
        input,
        nom::error::ErrorKind::SeparatedList,
    )));
}

fn array_values(input: &[u8]) -> IResult<&[u8], Vec<Value<'_>>> {
    delimited(
        terminated(char('['), multispace0),
        separated_list0(delimited(multispace0, char(','), multispace0), value),
        preceded(multispace0, char(']')),
    )(input)
}

fn key_value(input: &[u8]) -> IResult<&[u8], (Value<'_>, Value<'_>)> {
    map(
        separated_pair(
            string,
            delimited(multispace0, char(':'), multispace0),
            value,
        ),
        |(k, v)| (k, v),
    )(input)
}

fn object_values(input: &[u8]) -> IResult<&[u8], Vec<(Value<'_>, Value<'_>)>> {
    delimited(
        terminated(char('{'), multispace0),
        separated_list0(delimited(multispace0, char(','), multispace0), key_value),
        preceded(multispace0, char('}')),
    )(input)
}

// Parse JSON text to JSONB Value.
// Inspired by `https://github.com/jorgecarleitao/json-deserializer`
// Thanks Jorge Leitao.
pub fn parse_value(buf: &[u8]) -> Result<Value<'_>, Error> {
    let mut parser = Parser::new(buf);
    parser.parse()
}

// used to parse value from storage.
// as value has be parsed, string don't need extra escape.
pub fn decode_value(buf: &[u8]) -> Result<Value<'_>, Error> {
    let mut parser = Parser::new_with_escaped(buf);
    parser.parse()
}

struct Parser<'a> {
    buf: &'a [u8],
    idx: usize,
    escaped: bool,
}

impl<'a> Parser<'a> {
    fn new(buf: &'a [u8]) -> Parser<'a> {
        Self {
            buf,
            idx: 0,
            escaped: false,
        }
    }

    fn new_with_escaped(buf: &'a [u8]) -> Parser<'a> {
        Self {
            buf,
            idx: 0,
            escaped: true,
        }
    }

    fn parse(&mut self) -> Result<Value<'a>, Error> {
        let val = self.parse_json_value()?;
        self.skip_unused();
        if self.idx < self.buf.len() {
            self.step();
            return Err(self.error(ParseErrorCode::UnexpectedTrailingCharacters));
        }
        Ok(val)
    }

    fn parse_json_value(&mut self) -> Result<Value<'a>, Error> {
        self.skip_unused();
        let c = self.next()?;
        match c {
            b'n' => self.parse_json_null(),
            b't' => self.parse_json_true(),
            b'f' => self.parse_json_false(),
            b'0'..=b'9' | b'-' => self.parse_json_number(),
            b'"' => self.parse_json_string(),
            b'[' => self.parse_json_array(),
            b'{' => self.parse_json_object(),
            _ => {
                self.step();
                Err(self.error(ParseErrorCode::ExpectedSomeValue))
            }
        }
    }

    fn next(&mut self) -> Result<&u8, Error> {
        match self.buf.get(self.idx) {
            Some(c) => Ok(c),
            None => Err(self.error(ParseErrorCode::InvalidEOF)),
        }
    }

    fn must_is(&mut self, c: u8) -> Result<(), Error> {
        match self.buf.get(self.idx) {
            Some(v) => {
                self.step();
                if v == &c {
                    Ok(())
                } else {
                    Err(self.error(ParseErrorCode::ExpectedSomeIdent))
                }
            }
            None => Err(self.error(ParseErrorCode::InvalidEOF)),
        }
    }

    fn check_next(&mut self, c: u8) -> bool {
        if self.idx < self.buf.len() {
            let v = self.buf.get(self.idx).unwrap();
            if v == &c {
                return true;
            }
        }
        false
    }

    fn check_next_either(&mut self, c1: u8, c2: u8) -> bool {
        if self.idx < self.buf.len() {
            let v = self.buf.get(self.idx).unwrap();
            if v == &c1 || v == &c2 {
                return true;
            }
        }
        false
    }

    fn check_digit(&mut self) -> bool {
        if self.idx < self.buf.len() {
            let v = self.buf.get(self.idx).unwrap();
            if v.is_ascii_digit() {
                return true;
            }
        }
        false
    }

    fn step_digits(&mut self) -> Result<usize, Error> {
        if self.idx == self.buf.len() {
            return Err(self.error(ParseErrorCode::InvalidEOF));
        }
        let mut len = 0;
        while self.idx < self.buf.len() {
            let c = self.buf.get(self.idx).unwrap();
            if !c.is_ascii_digit() {
                break;
            }
            len += 1;
            self.step();
        }
        Ok(len)
    }

    #[inline]
    fn step(&mut self) {
        self.idx += 1;
    }

    #[inline]
    fn step_by(&mut self, n: usize) {
        self.idx += n;
    }

    fn error(&self, code: ParseErrorCode) -> Error {
        let pos = self.idx;
        Error::Syntax(code, pos)
    }

    fn skip_unused(&mut self) {
        while self.idx < self.buf.len() {
            let c = self.buf.get(self.idx).unwrap();
            if !matches!(c, b'\n' | b' ' | b'\r' | b'\t') {
                break;
            }
            self.step();
        }
    }

    fn parse_json_null(&mut self) -> Result<Value<'a>, Error> {
        let data = [b'n', b'u', b'l', b'l'];
        for v in data.into_iter() {
            self.must_is(v)?;
        }
        Ok(Value::Null)
    }

    fn parse_json_true(&mut self) -> Result<Value<'a>, Error> {
        let data = [b't', b'r', b'u', b'e'];
        for v in data.into_iter() {
            self.must_is(v)?;
        }
        Ok(Value::Bool(true))
    }

    fn parse_json_false(&mut self) -> Result<Value<'a>, Error> {
        let data = [b'f', b'a', b'l', b's', b'e'];
        for v in data.into_iter() {
            self.must_is(v)?;
        }
        Ok(Value::Bool(false))
    }

    fn parse_json_number(&mut self) -> Result<Value<'a>, Error> {
        let start_idx = self.idx;

        let mut has_fraction = false;
        let mut has_exponent = false;
        let mut negative: bool = false;

        if self.check_next(b'-') {
            negative = true;
            self.step();
        }
        if self.check_next(b'0') {
            self.step();
            if self.check_digit() {
                self.step();
                return Err(self.error(ParseErrorCode::InvalidNumberValue));
            }
        } else {
            let len = self.step_digits()?;
            if len == 0 {
                self.step();
                return Err(self.error(ParseErrorCode::InvalidNumberValue));
            }
        }
        if self.check_next(b'.') {
            has_fraction = true;
            self.step();
            let len = self.step_digits()?;
            if len == 0 {
                self.step();
                return Err(self.error(ParseErrorCode::InvalidNumberValue));
            }
        }
        if self.check_next_either(b'E', b'e') {
            has_exponent = true;
            self.step();
            if self.check_next_either(b'+', b'-') {
                self.step();
            }
            let len = self.step_digits()?;
            if len == 0 {
                self.step();
                return Err(self.error(ParseErrorCode::InvalidNumberValue));
            }
        }
        let s = unsafe { std::str::from_utf8_unchecked(&self.buf[start_idx..self.idx]) };

        if !has_fraction && !has_exponent {
            if !negative {
                if let Ok(v) = s.parse::<u64>() {
                    return Ok(Value::Number(Number::UInt64(v)));
                }
            } else if let Ok(v) = s.parse::<i64>() {
                return Ok(Value::Number(Number::Int64(v)));
            }
        }

        match fast_float::parse(s) {
            Ok(v) => Ok(Value::Number(Number::Float64(v))),
            Err(_) => Err(self.error(ParseErrorCode::InvalidNumberValue)),
        }
    }

    fn parse_json_string(&mut self) -> Result<Value<'a>, Error> {
        self.must_is(b'"')?;

        let start_idx = self.idx;
        let mut escapes = 0;
        loop {
            let c = self.next()?;
            if c.is_ascii_control() {
                return Err(self.error(ParseErrorCode::ControlCharacterWhileParsingString));
            }
            match c {
                b'\\' => {
                    self.step();
                    escapes += 1;
                    let next_c = self.next()?;
                    if *next_c == b'u' {
                        self.step_by(UNICODE_LEN + 1);
                    } else {
                        self.step();
                    }
                    continue;
                }
                b'"' => {
                    self.step();
                    break;
                }
                _ => {}
            }
            self.step();
        }

        let mut data = &self.buf[start_idx..self.idx - 1];
        let val = if !self.escaped && escapes > 0 {
            let len = self.idx - 1 - start_idx - escapes;
            let mut idx = start_idx + 1;
            let mut str_buf = String::with_capacity(len);
            while !data.is_empty() {
                idx += 1;
                let byte = data[0];
                if byte == b'\\' {
                    data = &data[1..];
                    data = parse_escaped_string(data, &mut idx, &mut str_buf)?;
                } else {
                    str_buf.push(byte as char);
                    data = &data[1..];
                }
            }
            Cow::Owned(str_buf)
        } else {
            std::str::from_utf8(data)
                .map(Cow::Borrowed)
                .map_err(|_| self.error(ParseErrorCode::InvalidStringValue))?
        };
        Ok(Value::String(val))
    }

    fn parse_json_array(&mut self) -> Result<Value<'a>, Error> {
        self.must_is(b'[')?;

        let mut first = true;
        let mut values = Vec::new();
        loop {
            self.skip_unused();
            let c = self.next()?;
            if *c == b']' {
                self.step();
                break;
            }
            if !first {
                if *c != b',' {
                    return Err(self.error(ParseErrorCode::ExpectedArrayCommaOrEnd));
                }
                self.step();
            }
            first = false;
            let value = self.parse_json_value()?;
            values.push(value);
        }
        Ok(Value::Array(values))
    }

    fn parse_json_object(&mut self) -> Result<Value<'a>, Error> {
        self.must_is(b'{')?;

        let mut first = true;
        let mut obj = Object::new();
        loop {
            self.skip_unused();
            let c = self.next()?;
            if *c == b'}' {
                self.step();
                break;
            }
            if !first {
                if *c != b',' {
                    return Err(self.error(ParseErrorCode::ExpectedObjectCommaOrEnd));
                }
                self.step();
            }
            first = false;
            let key = self.parse_json_value()?;
            if !key.is_string() {
                return Err(self.error(ParseErrorCode::KeyMustBeAString));
            }
            self.skip_unused();
            let c = self.next()?;
            if *c != b':' {
                return Err(self.error(ParseErrorCode::ExpectedColon));
            }
            self.step();
            let value = self.parse_json_value()?;

            let k = key.as_str().unwrap();
            obj.insert(k.to_string(), value);
        }
        Ok(Value::Object(obj))
    }
}
