use std::collections::HashMap;
use ordered_float::OrderedFloat;

pub fn histogram(values: impl Iterator<Item = f64>) -> (HashMap<OrderedFloat<f64>, usize>, usize) {
	let mut histogram = HashMap::new();
	let mut len = 0;

	for value in values {
		*histogram.entry(OrderedFloat(value)).or_default() += 1;
		len += 1;
	}

	(histogram, len)
}

pub fn gini_val(histogram: &HashMap<OrderedFloat<f64>, usize>, len: usize) -> f64 {
	1.0 - histogram
		.iter()
		.map(|(_, &n)| (n as f64 / len as f64).powi(2))
		.sum::<f64>()
}

pub fn gini(values: impl Iterator<Item = f64>) -> f64 {
	let (histogram, len) = histogram(values);
	gini_val(&histogram, len)
}

pub fn most_frequent(values: impl Iterator<Item = f64>) -> f64 {
	let (histogram, _) = histogram(values);

	histogram
		.into_iter()
		.max_by_key(|&(_, v)| v)
		.map(|(k, _)| k.into_inner())
		.unwrap()
}