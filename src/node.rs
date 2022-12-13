use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Write};

const LEAF: u16 = 0;
const CHILDREN: u16 = 1;

#[derive(Debug)]
pub struct Split {
	pub value: f64,
	pub column: usize,
}

impl Split {
	pub fn serialize<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
		writer.write_f64::<BigEndian>(self.value)?;
		writer.write_u16::<BigEndian>(self.column as u16)?;

		Ok(())
    }

	pub fn deserialize<R: Read>(reader: &mut R) -> std::io::Result<Self> {
		let value = reader.read_f64::<BigEndian>()?;
		let column = reader.read_u16::<BigEndian>()? as usize;

        Ok(Self { value, column })
    }
}

#[derive(Debug)]
pub enum Node {
	Leaf(f64),
	Children {
		left: Box<Node>,
		right: Box<Node>,
		split: Split,
	},
}

impl Node {
	pub fn predict(&self, x: &[f64]) -> f64 {
		match &self {
			Node::Leaf(value) => *value,
			Node::Children { left, right, split } => {
				if x[split.column] < split.value {
					left.predict(x)
				} else {
					right.predict(x)
				}
			},
		}
	}

	pub fn serialize<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
		match &self {
			Node::Leaf(value) => {
				writer.write_u16::<BigEndian>(LEAF)?;
				writer.write_f64::<BigEndian>(*value)?;
			},
			Node::Children { left, right, split } => {
				writer.write_u16::<BigEndian>(CHILDREN)?;
				split.serialize(writer)?;
				left.serialize(writer)?;
				right.serialize(writer)?;
			}
		}

		Ok(())
    }

	pub fn deserialize<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        match reader.read_u16::<BigEndian>()? {
			LEAF => Ok(Node::Leaf(reader.read_f64::<BigEndian>()?)),
			CHILDREN => {
				let split = Split::deserialize(reader)?;
				let left = Box::new(Node::deserialize(reader)?);
				let right = Box::new(Node::deserialize(reader)?);

				Ok(Node::Children { split, left, right })
			},
			i => Err(std::io::Error::new(
				std::io::ErrorKind::InvalidData,
				format!("unknown tree type {:?}", i),
			)),
		}
    }
}