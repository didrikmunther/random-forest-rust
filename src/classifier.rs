use std::io::{Read, Write};

pub trait Classifier: Sized {
	fn predict(&self, x: &[f64]) -> f64;

	fn serialize<W: Write>(&self, writer: &mut W) -> std::io::Result<()>;
	fn deserialize<R: Read>(reader: &mut R) -> std::io::Result<Self>;
}