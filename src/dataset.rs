use crate::classifier::Classifier;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::ops::Range;
use rand::seq::SliceRandom;
use rand::Rng;
use ordered_float::OrderedFloat;

#[derive(Clone, Debug)]
pub struct Dataset<'a> {
	columns: &'a [Vec<f64>],
	targets: &'a [f64],

	index: Vec<usize>,
    range: Range<usize>,
}

impl<'a, 'b> Dataset<'a> {
	pub fn evaluate<C: Classifier + Sync>(&self, classifier: &C) -> f64 {
		self.rows()
			.zip(self.targets())
			.collect::<Vec<_>>()
			.into_par_iter()
			.filter(|(x, y)| classifier.predict(&x) == *y)
			.count() as f64 / self.rows_len() as f64
	}

	pub fn sort(&mut self, column: usize) {
		(&mut self.index[self.range.start..self.range.end])
			.sort_by_key(|&x| OrderedFloat(self.columns[column][x]));
	}

	pub fn get_splits(&'b self, column: usize) -> impl 'b + Iterator<Item = (Range<usize>, f64)> {
		let column = &self.columns[column];

		self.indices()
			.map(|x| column[x])
			.enumerate()
			.scan(None, |prev, (i, x)| {
				if prev.is_none() {
					*prev = Some((x, i));
					Some(None)
				} else if prev.map_or(false, |(y, _)| (y - x).abs() > std::f64::EPSILON) {
					let (y, _) = prev.unwrap();
					*prev = Some((x, i));

					Some(Some((
						0..i,
						(x + y) / 2.0
					)))
				} else {
					Some(None)
				}
			})
			.filter_map(|t| t)
	}

	pub fn split<F, T>(&mut self, row: usize, mut f: F) -> (T, T)
	where
		F: FnMut(&mut Self) -> T,
	{
		let row = row + self.range.start;
		let original = self.range.clone();

        self.range.end = row;
        let left = f(self);
        self.range.end = original.end;

        self.range.start = row;
        let right = f(self);
        self.range.start = original.start;

		(left, right)
	}

    pub fn train_test_split<R: Rng + ?Sized>(mut self, rng: &mut R, test_rate: f64) -> (Self, Self) {
        (&mut self.index[self.range.start..self.range.end]).shuffle(rng);
        let test_num = (self.rows_len() as f64 * test_rate).round() as usize;

        let mut train = self.clone();
        let mut test = self;
        test.range.end = test.range.start + test_num;
        train.range.start = test.range.end;

        (train, test)
    }

	pub fn bootstrap<R: Rng + ?Sized>(&self, rng: &mut R, max_samples: usize) -> Self {
		let samples = std::cmp::min(max_samples, self.rows_len());

		let range = 0..samples;
        let index = range
			.clone()
            .map(|_| self.index[rng.gen_range(self.range.start, self.range.end)])
            .collect::<Vec<_>>();

		// Very performance intense to calculate oob

		// // Get the indexes which did not end up in the boostrap
		// let oob_index = self.index[self.range.start..self.range.end].iter()
		// 	.filter(|&x| !index.contains(x))
		// 	.map(|&x| x)
		// 	.collect::<Vec<_>>();

		// let oob = Self {
		// 	range: 0..oob_index.len(),
		// 	index: oob_index,
		// 	columns: self.columns,
		// 	targets: self.targets
		// };

        Self {
            index,
            range,
            columns: self.columns,
			targets: self.targets
        }
	}

	fn indices(&'b self) -> impl 'b + Iterator<Item = usize> + Clone {
		self.index[self.range.start..self.range.end]
			.iter()
			.map(|&x| x)
	}

	pub fn targets(&'b self) -> impl 'b + Iterator<Item = f64> {
		self.indices()
			.map(|i| self.targets[i])
	}

	pub fn column(&'b self, column: usize) -> impl 'b + Iterator<Item = f64> {
		let column = &self.columns[column];

		self.indices()
            .map(|i| column[i])
	}

	pub fn features_len(&self) -> usize {
		return self.columns.len()
	}

	pub fn rows_len(&self) -> usize {
        self.range.end - self.range.start
    }

	pub fn rows(&'b self) -> impl 'b + Iterator<Item = Vec<f64>> {
		self.indices().map(move |i| {
            (0..self.columns.len())
                .map(|j| self.columns[j][i])
                .collect()
        })
	}

	pub fn classify<C: Classifier + Sized>(&self, classifier: &C) -> Vec<f64> {
		self.rows()
			.map(|x| classifier.predict(&x))
			.collect()
	}
}

#[derive(Debug)]
pub struct Builder {
	columns: Vec<Vec<f64>>,
	targets: Vec<f64>,
}

impl Builder {
	pub fn new() -> Self {
		Self {
			columns: Vec::new(),
			targets: Vec::new(),
		}
	}

	pub fn build(&self) -> Dataset {
		let range = 0..self.targets.len();

		Dataset {
			columns: &self.columns,
			targets: &self.targets,

			range: range.clone(),
			index: range.collect(),
		}
	}

	pub fn add(&mut self, x: &[f64], y: f64) {
		if self.columns.is_empty() {
			self.columns = vec![Vec::new(); x.len()];
		}

		for (column, value) in self.columns.iter_mut().zip(x) {
			column.push(*value);
		}

		self.targets.push(y);
	}

	pub fn add_x(&mut self, x: &[f64]) {
		if self.columns.is_empty() {
			self.columns = vec![Vec::new(); x.len()];
		}

		for (column, value) in self.columns.iter_mut().zip(x) {
			column.push(*value)
		}

		self.targets.push(0.0);
	}
}