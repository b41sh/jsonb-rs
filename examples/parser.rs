// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.


use std::fs;

//use std::fs::File;
//use std::io::{BufReader, Seek};
use std::time::Instant;

use jsonb::new_parser::Parser;
use jsonb::Error;
use jsonb::to_string;
use jsonb::to_pretty_string;

// cargo run --example parser --release
fn main() -> Result<(), Error> {
    use std::env;
    let args: Vec<String> = env::args().collect();
/**
     let s = r#"{"abcd":12.34,                                                 "xxaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaxxxaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaxxxaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaxxxaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaxxxaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaxxxaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaxxxaaaaaaaaaaaaaaaaaaaaaaaaaaaaxxaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaxxxaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaxaaaaaaaaaxxxaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaxxxaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaax":444.55, "glossary":{"title":"example glossary","GlossDiv":{"title":"S","GlossList":            {"GlossEntry":{"ID":"SGML","SortAs":"SGML","GlossTerm":"Standard Generalized Markup Language","Acronym":        "SGML","Abbrev":"ISO 8879:1986","GlossDef":{"para":"A meta-markup language, used to create markup languages     such as DocBook.","GlossSeeAlso":["GML","XML"]},"GlossSee":"markup"}}}}}"#.to_string();

    println!("s={:?}", s);
    let data = s.into_bytes();

    let mut parser = Parser::new();
    let buf = parser.parse(&data).unwrap();
    println!("buf={:?}", buf);

    let s = to_string(&buf);
    println!("\ns={}", s);
    let ss = to_pretty_string(&buf);
    println!("\nss={}", ss);

    println!("\n\n\n");
*/

    let mut parser = Parser::new();

    let file_path = &args[1];
    let mut c = fs::read(file_path).unwrap();

    let t = Instant::now();
    println!("t={:?}", t);

    let buf = parser.parse(&c).unwrap();
    //println!("buf={:?}", buf);

    let t2 = Instant::now();
    println!("t2={:?}", t2);

    //let s = to_string(&buf);
    //println!("\ns={}", s);
    //let ss = to_pretty_string(&buf);
    //println!("\nss={}", ss);

    println!("cost {:?} ms", t.elapsed().as_millis());

    let t3 = Instant::now();
    let _sv = simd_json::to_borrowed_value(&mut c).unwrap();
    //println!("simd v={:?}", sv);
    println!("simd cost {:?} ms", t3.elapsed().as_millis());


    Ok(())
}
