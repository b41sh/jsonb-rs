use core::fmt::Debug;
use core::mem::ManuallyDrop;
use std::collections::BTreeMap;

use crate::jentry::JEntry;
use crate::builder::ArrayBuilder;
use crate::builder::ObjectBuilder;
use crate::builder::Entry;

use crate::new_parser::ParseError;

use crate::Number;
use simd_json::Tape;
use simd_json::Node;
use simd_json::StaticNode;

pub fn nparse(data: &mut [u8]) -> Result<Vec<u8>, ParseError> {
    let tape = simd_json::to_tape(data).unwrap();


    let mut buf = Vec::new();
    let (entry, _) = stage3(0, &tape.0)?;

    match entry {
        Entry::ArrayBuilder(builder) => {
            builder.build_into(&mut buf);
        }
        Entry::ObjectBuilder(builder) => {
            builder.build_into(&mut buf);
        }
        _ => {},
    }
    Ok(buf)
}

fn stage3<'a>(i: usize, nodes: &'a Vec<Node>) -> Result<(Entry<'a>, usize), ParseError> {
    //println!("nodes={:?}", nodes);
    let node = unsafe { nodes.get_unchecked(i) };
    let (entry, count) = match node {
        Node::Object { len, count } => {
            let mut idx = i + 1;
            let mut obj_builder = ObjectBuilder::new();
            for _ in 0..*len {
                let key_node = unsafe { nodes.get_unchecked(idx) };
                let key = match key_node {
                    Node::String(s) => s,
                    _ => unreachable!(),
                };
                idx += 1;
                let (entry, inner_count) = stage3(idx, nodes)?;
                match entry {
                    Entry::ArrayBuilder(builder) => {
                        obj_builder.push_array(key, builder);
                    }
                    Entry::ObjectBuilder(builder) => {
                        obj_builder.push_object(key, builder);
                    }
                    Entry::Number(num) => {
                        obj_builder.push_number(key, num);
                    }
                    Entry::Raw(jentry, val) => {
                        obj_builder.push_raw(key, jentry, val);
                    }
                }
                idx += inner_count;
            }
            //println!("object len={:?} count={:?}", len, count);
            (Entry::ObjectBuilder(obj_builder), *count + 1)
        }
        Node::Array { len, count } => {
            //println!("array len={:?} count={:?}", len, count);
            let mut idx = i + 1;
            let mut arr_builder = ArrayBuilder::new(*len);
            for _ in 0..*len {
                let (entry, inner_count) = stage3(idx, nodes)?;
                match entry {
                    Entry::ArrayBuilder(builder) => {
                        arr_builder.push_array(builder);
                    }
                    Entry::ObjectBuilder(builder) => {
                        arr_builder.push_object(builder);
                    }
                    Entry::Number(num) => {
                        arr_builder.push_number(num);
                    }
                    Entry::Raw(jentry, val) => {
                        arr_builder.push_raw(jentry, val);
                    }
                }
                idx += inner_count;
            }
            (Entry::ArrayBuilder(arr_builder), *count + 1)
        }
        Node::Static(static_node) => {
            //let entry = Entry::Raw(JEntry::make_null_jentry(), &[]);
            //println!("static_node={:?}", static_node);
            let entry = match static_node {
                StaticNode::Null => {
                    Entry::Raw(JEntry::make_null_jentry(), &[])
                }
                StaticNode::Bool(v) => {
                    if *v {
                        Entry::Raw(JEntry::make_true_jentry(), &[])
                    } else {
                        Entry::Raw(JEntry::make_false_jentry(), &[])
                    }
                }
                StaticNode::I64(n) => {
                    Entry::Number(Number::Int64(*n))
                }
                StaticNode::U64(n) => {
                    Entry::Number(Number::UInt64(*n))
                }
                StaticNode::F64(n) => {
                    Entry::Number(Number::Float64(*n))
                }
            };
            (entry, 1)
        }
        Node::String(s) => {
            let entry = Entry::Raw(JEntry::make_string_jentry(s.len()), s.as_bytes());
            (entry, 1)
        }
    };
    Ok((entry, count))
}
