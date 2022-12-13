use std::mem;
use std::fs;
use std::io::{self, BufRead};
use crate::dataset;

pub fn read_evaluation(dataset_location: &str) -> std::io::Result<dataset::Builder> {
	let file = fs::File::open(dataset_location)?;
	let lines = io::BufReader::new(file).lines();

	let mut builder = dataset::Builder::new();
	let mut size = 0;

	for (i, line) in lines.enumerate() {
		if let Ok(x) = line {
			let row = x
				.split(",")
				.map(|x| x.parse::<f64>().unwrap())
				.collect::<Vec<f64>>();

			builder.add_x(&row);

			size += mem::size_of_val(&row);
			if i % 1000 == 0 {
				print!("\r[{:.2}%] {:.1}MB        ", i as f64 / 700000 as f64 * 100.0, size as f64 / 1e6);
			}
		}
	}

	println!("\r[100%] {:.1}MB        ", size as f64 / 1e6);

	Ok(builder)
}

pub fn read(take: usize, dataset_location: &str) -> std::io::Result<dataset::Builder> {
	let file = fs::File::open(dataset_location)?;
	let lines = io::BufReader::new(file).lines().take(take);

	let mut builder = dataset::Builder::new();
	let mut size = 0;

	for (i, line) in lines.enumerate() {
		if let Ok(x) = line {
			let row = x
				.split(",")
				.map(|x| x.parse::<f64>().unwrap())
				.collect::<Vec<f64>>();

			let (y, x) = row.split_last().unwrap();
			builder.add(x, *y);

			size += mem::size_of_val(x);
			if i % 1000 == 0 {
				print!("\r[{:.2}%] {:.1}MB        ", i as f64 / take as f64 * 100.0, size as f64 / 1e6);
			}
		}
	}

	println!("\r[100%] {:.1}MB        ", size as f64 / 1e6);

	Ok(builder)
}