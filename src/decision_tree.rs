use std::io::{Read, Write};
use std::collections::HashMap;
use rand::Rng;
use rand::seq::SliceRandom as _;
use ordered_float::OrderedFloat;

use crate::dataset::Dataset;
use crate::node::{Node, Split};
use crate::classifier::Classifier;
use crate::functions::{self, gini, most_frequent, gini_val};

// Sliding window of gini
// https://arxiv.org/pdf/1403.6348.pdf
struct SlidingGini {
	n: usize,
	ni: HashMap<OrderedFloat<f64>, usize>,
	g: f64
}

impl SlidingGini {
	pub fn new(targets: impl Iterator<Item = f64>) -> Self {
		let (histogram, len) = functions::histogram(targets);

		Self {
			n: len,
			g: gini_val(&histogram, len),
			ni: histogram,
		}
	}

	pub fn inc(&mut self, typ: OrderedFloat<f64>) {
		let entry = self.ni.entry(typ).or_insert(0);
		*entry += 1;
		self.n += 1;
		self.g = 1.0 - 1.0 / (self.n as f64).powi(2) * (((self.n - 1) as f64).powi(2) * (1.0 - self.g) + 2.0 * *entry as f64 - 1.0);
	}

	pub fn dec(&mut self, typ: OrderedFloat<f64>) {
		let entry = self.ni.entry(typ).or_insert(0);
		*entry -= 1;
		self.n -= 1;
		self.g = 1.0 - 1.0 / (self.n as f64).powi(2) * (((self.n + 1) as f64).powi(2) * (1.0 - self.g) - 2.0 * *entry as f64 - 1.0);
	}

	pub fn gini(&self) -> f64 {
		self.g
	}
}

struct NodeBuilder<R> {
	max_features: usize,
	max_depth: usize,
	rng: R
}

impl<R: Rng> NodeBuilder<R> {
	fn build(&mut self, dataset: &mut Dataset, depth: usize) -> Node {
		if depth > self.max_depth {
			return Node::Leaf(most_frequent(dataset.targets()));
		}

		let impurity = gini(dataset.targets());

		let mut best_split: Option<Split> = None;
		let mut best_gain = std::f64::MIN;
		let columns = (0..dataset.features_len()).enumerate().map(|(i, _)| i).collect::<Vec<usize>>();
		let max_features = std::cmp::min(columns.len(), self.max_features);

		for &column in columns.choose_multiple(&mut self.rng, max_features) {
			dataset.sort(column);

			let mut left_window: Option<SlidingGini> = None;
			let mut right_window: Option<SlidingGini> = None;

			let mut prev_range = 0..0;
			let mut targets = dataset.targets();

            for (left, value) in dataset.get_splits(column) {
				// Old way of calculating gini value
				// let rows_l = dataset.targets().take(left.end).skip(left.start);
				// let rows_r = dataset.targets().skip(left.end);
				// let impurity_l = gini(rows_l);
                // let impurity_r = gini(rows_r);

				// For the first time, we want to initialize the windows.
				// But we cannot break just after, because then the targets iterator
				//  will not be in sync with our split.
				let mut do_inc = true;
				for _ in prev_range.end..left.end {
					let cls = OrderedFloat(targets.next().unwrap());

					if do_inc {
						if let Some(window) = &mut left_window {
							window.inc(cls);

							if let Some(window) = &mut right_window {
								window.dec(cls);
							}
						} else {
							left_window = Some(SlidingGini::new(dataset.targets().take(left.end).skip(left.start)));
							right_window = Some(SlidingGini::new(dataset.targets().skip(left.end)));
							do_inc = false;
						}
					}
				}

				let impurity_l = left_window.as_ref()
					.map(SlidingGini::gini)
					.unwrap_or_else(|| gini(dataset.targets()
							.take(left.end)
							.skip(left.start)));

                let impurity_r = right_window.as_ref()
					.map(SlidingGini::gini)
					.unwrap_or_else(|| gini(dataset.targets()
						.skip(left.end)));

				let ratio_l = (left.end - left.start) as f64 / dataset.rows_len() as f64;
                let ratio_r = 1.0 - ratio_l;

				let gain = impurity - (ratio_l * impurity_l + ratio_r * impurity_r);
				
                if best_gain < gain {
					best_split = Some(Split { column, value });
                    best_gain = gain;
                }

				prev_range = left;
			}
		}
		
		if let Some(split) = best_split {
            self.build_children(dataset, split, depth)
        } else {
            Node::Leaf(most_frequent(dataset.targets()))
        }
	}

	pub fn build_children(&mut self, dataset: &mut Dataset, split: Split, depth: usize) -> Node {
		dataset.sort(split.column);

		let split_row = dataset
            .column(split.column)
            .take_while(|&f| f <= split.value)
            .count();

		let (left, right) = dataset.split(split_row, |x| Box::new(self.build(x, depth + 1)));

		Node::Children {
			left, right, split
		}
	}
}

pub struct DecisionTree {
	root: Node,
}

impl Classifier for DecisionTree {
	fn predict(&self, x: &[f64]) -> f64 {
		self.root.predict(x)
	}

	fn serialize<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
		self.root.serialize(writer)
    }

	fn deserialize<R: Read>(reader: &mut R) -> std::io::Result<Self> {
		let root = Node::deserialize(reader)?;

        Ok(Self {
			root
		})
    }
}

pub struct DecisionTreeBuilder {
	pub max_features: Option<usize>,
	pub max_depth: usize,
}

impl Default for DecisionTreeBuilder {
	fn default() -> Self {
		Self {
			max_features: None,
			max_depth: 32,
		}
	}
}

impl DecisionTreeBuilder {
	pub fn fit<R: Rng + ?Sized>(&self, rng: &mut R, mut dataset: Dataset) -> DecisionTree {
		let max_features = self.max_features.unwrap_or(dataset.features_len());
		let root = (NodeBuilder {
			max_features,
			max_depth: self.max_depth,
			rng
		}).build(&mut dataset, 1);

		DecisionTree { root }
	}
}

