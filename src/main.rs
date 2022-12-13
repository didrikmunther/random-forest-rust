extern crate num_derive;

use rand::{SeedableRng, rngs::StdRng, Rng, thread_rng};
use std::collections::BTreeMap;
use ordered_float::OrderedFloat;
use std::time::{Instant};
use std::fs::File;
use std::io::{Write};
use random_forest::{RandomForestBuilder, RandomForestClassifier};

use classifier::Classifier;

mod challenge_data;
mod dataset;
mod decision_tree;
mod random_forest;
mod iris;
mod functions;
mod classifier;
mod node;

#[allow(dead_code)]
fn train<R: Rng + ?Sized>(builder: &dataset::Builder, classifier_builder: &RandomForestBuilder, rng: &mut R, split: f64) -> (RandomForestClassifier, f64, f64) {
	println!("Building dataset ...");
	let dataset = builder.build();

	print!("Splitting dataset to train and test: ");
	let (train, test) = dataset.train_test_split(rng, 1.0 - split);
	println!(" test: {}, train: {}", test.rows_len(), train.rows_len());
	
	println!("Fitting random forest classifier [tree: {}, depth: {}] ...", classifier_builder.n_trees, classifier_builder.max_depth);
	let classifier = classifier_builder.fit(train.clone());

	// let classifier = decision_tree::DecisionTreeBuilder::default().fit(&mut rng, train);

	println!("Evaluating classifier ...");
	let test_rate = test.evaluate(&classifier);
	let train_rate = train.evaluate(&classifier);

	println!("Trained on {} samples, tested on {} samples ({:.4}% test)", train.rows_len(), test.rows_len(), (1.0 - split) * 100.0);
	println!("Classification rate test: {:.3?}%, train: {:.3?}%", test_rate * 100.0, train_rate * 100.0);

	(classifier, test_rate, train_rate)
}

#[allow(dead_code)]
fn try_serialization() -> std::io::Result<()> {
	let mut rng: StdRng = SeedableRng::from_seed([0u8; 32]);

	let builder = iris::load_iris();
	let dataset = builder.build();
	let (train, test) = dataset.train_test_split(&mut rng, 0.5);

	let rforest = RandomForestBuilder {
		n_trees: 100,
		max_depth: 64,
		bag_amount: 0.5,
	};
	let classifier = rforest.fit(train.clone());

	let test_rate = test.evaluate(&classifier);
	let train_rate = train.evaluate(&classifier);

	println!("Trained on {} samples, tested on {} samples ({}% test)", train.rows_len(), test.rows_len(), 0.5 * 100.0);
	println!("Classification rate test: {:.3?}%, train: {:.3?}%", test_rate * 100.0, train_rate * 100.0);

	println!("Serializing ...");

	let mut file = File::create("serialized.classifier")?;
	classifier.serialize(&mut file)?;

	println!("Serialized!");
	println!("Deserializing ...");

	let mut file = File::open("serialized.classifier")?;
	let classifier = RandomForestClassifier::deserialize(&mut file)?;

	println!("Deserialized!");

	let test_rate = test.evaluate(&classifier);
	let train_rate = train.evaluate(&classifier);

	println!("Trained on {} samples, tested on {} samples ({}% test)", train.rows_len(), test.rows_len(), 0.5 * 100.0);
	println!("Classification rate test: {:.3?}%, train: {:.3?}%", test_rate * 100.0, train_rate * 100.0);

	Ok(())
}

#[derive(Debug, Copy, Clone)]
#[allow(dead_code)]
struct TrainResult {
	train: f64,
	test: f64
}

#[allow(dead_code)]
fn bench(results_file: &str, splits: &[f64], classifier_builder: &RandomForestBuilder, do_serialize: bool) -> std::io::Result<BTreeMap<OrderedFloat<f64>, TrainResult>> {
	let mut rng: StdRng = SeedableRng::from_seed([0u8; 32]);

	let start = Instant::now();

	let args = std::env::args().collect::<Vec<String>>();
	let _split = 1.0 - args.get(1).and_then(|x| x.parse::<f64>().ok()).unwrap_or_else(|| 10000.0 / 700000.0 * 100.0) / 100.0;
	let take = args.get(2).and_then(|x| x.parse::<usize>().ok()).unwrap_or_else(|| 700000);

	println!("Reading dataset ...");
	let builder = challenge_data::read(take, "./input.dataset")?;
	let rows_len = builder.build().rows_len();

	let mut results: BTreeMap<OrderedFloat<f64>, TrainResult> = BTreeMap::new();
	let mut file = File::create(results_file)?;

	for &split in splits {
		let train_time = Instant::now();
		let (classifier, test, train) = train(&builder, classifier_builder, &mut rng, split);
		results.insert(OrderedFloat(split), TrainResult {
			train,
			test
		});

		let line = format!(
			"Train:test ratio: {:.2}%, train amount: {} | Test: {:.4}%, train: {:.4}% | time: {:.2} m\n",
			split * 100.0,
			(split * rows_len as f64).ceil() as i32,
			test * 100.0,
			train * 100.0,
			train_time.elapsed().as_secs_f64() / 60.0
		);

		write!(&mut file, "{}", line)?;
		print!("{}", line);
	
		println!("For a samples size of {}", std::cmp::min(take, rows_len));
		println!("Total time: {:.2} m", start.elapsed().as_secs_f64() / 60.0);

		if do_serialize {
			let serialized_name = format!("./serialized/{:.5}.serialized", split * 100.0);
			println!("Saving result to \"{}\"", serialized_name);
			let mut serialized_file = File::create(serialized_name)?;
			classifier.serialize(&mut serialized_file)?;
		}
	}

	Ok(results)
}

#[allow(dead_code)]
fn try_backup(name: &str) -> std::io::Result<()> {
	println!("Loading dataset");
	let builder = challenge_data::read(1000000, "./input.dataset")?;
	let dataset = builder.build();

	println!("Reading serialized file");
	let mut file = File::open(name)?;

	println!("Deserializing");
	let classifier = random_forest::RandomForestClassifier::deserialize(&mut file)?;

	println!("Evaluating data");
	let test_rate = dataset.evaluate(&classifier);

	// This will have significantly higher test rate, since there is training data in this as well.
	// So this is not valid, it's just to check if the serialization works.
	println!("Classification rate test: {:.3?}%", test_rate * 100.0);
	println!("Backup good?");

	Ok(())
}

#[allow(dead_code)]
fn bench_grid() -> std::io::Result<()> {
	// Training splits
	// On my computer, seems to follow `time in minutes = split^1.6`
	// Inverse function, `split(time) = time^(1/1.6)`. Setting for 8 hours split(60*8) = 47.4%
	// A split of 70% would take 14.9 hours on my computer.
	// A split of 12% would take 1 hour.
	//! All above was pre-SlidingGini, will test performance after big train is done.
	// At a glance, seems to follow a significantly lower exponent function.
	// 
	// let splits = &[
	// 	0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008,
		// 0.009, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05,
		// 0.07, 0.1, 0.115, 0.12, 0.17, 0.2, 0.22, 0.27, 0.29, 0.33,
		// 0.71
	// ];

	// let depth_test = &[0.001, 0.002];
	// let depths: &[usize] = &[1, 2, 3];
	// let trees: &[usize] = &[30, 40];

	let depth_test = &[0.15, 0.2, 0.25];
	let depths: &[usize] = &[12];
	let trees: &[usize] = &[100, 150, 200];
	let bag_amounts: &[f64] = &[0.1, 0.3, 0.5, 0.7, 1.0];

	let mut results = BTreeMap::<(usize, usize, OrderedFloat<f64>), TrainResult>::new();

	// Bench grid
	for &depth in depths {
		for &tree in trees {
			for &bag_amount in bag_amounts {
				let file = format!("./benches/results_depth_{}_trees_{}_bag_{}.txt", depth, tree, bag_amount);

				let result = bench(&file, depth_test, &RandomForestBuilder {
					n_trees: tree,
					max_depth: depth,
					bag_amount: bag_amount,
				}, false)?;

				results.insert((tree, depth, OrderedFloat(bag_amount)), *result.get(&OrderedFloat(*depth_test.last().unwrap())).unwrap());
			}
		}
	}

	// Sort the results by the test classification rate
	let mut results = Vec::from_iter(results.into_iter());
	results.sort_by_key(|(_, v)| -OrderedFloat(v.test));

	// Print the results to file
	let mut file = File::create("benches/results_final.txt")?;
	write!(&mut file, "Results:")?;
	for (k, v) in results.iter() {
		let (tree, depth, bag) = k;
		write!(&mut file, "Tree: {tree}, depth: {depth}, bag: {bag}. Rate: {:?}\n", v)?;
	}

	Ok(())
}

// Best hyper parameters:
// tree: 200, depth: 12, bag: 0.5

fn fit(dataset_location: &str, serializing_location: &str) -> std::io::Result<()> {
	// let mut rng: StdRng = SeedableRng::from_seed([0u8; 32]);
	let mut rng = thread_rng();

	println!("Reading dataset ...");
	let builder = challenge_data::read(1000000, dataset_location)?;
	let dataset = builder.build();
	let (train, test) = dataset.train_test_split(&mut rng, 0.3);

	println!("Fitting model ...");
	let model = (RandomForestBuilder {
		n_trees: 200,
		max_depth: 12,
		bag_amount: 0.5,
	}).fit(train.clone());

	println!("Evaluating model ...");
	println!("Classification rate test: {:.3?}%", test.evaluate(&model) * 100.0);
	println!("Classification rate train: {:.3?}%", train.evaluate(&model) * 100.0);
	
	println!("Serializing model to {} ...", serializing_location);
	let mut serialized_file = File::create(serializing_location)?;
	model.serialize(&mut serialized_file)?;

	Ok(())
}

fn evaluate(dataset_location: &str, serializing_location: &str, output_location: &str) -> std::io::Result<()> {
	println!("Reading evaluation dataset ...");
	let builder = challenge_data::read_evaluation(dataset_location)?;
	let dataset = builder.build();

	println!("Reading serialized model {} ...", serializing_location);
	let mut file = File::open(serializing_location)?;

	println!("Deserializing model ...");
	let classifier = random_forest::RandomForestClassifier::deserialize(&mut file)?;

	println!("Classifying evaluation data ...");
	let classified = dataset.classify(&classifier);

	println!("Writing classified data to output file {} ...", output_location);
	let mut classified_file = File::create(output_location)?;
	
	for class in classified {
		writeln!(&mut classified_file, "{}", class)?;
	}

	Ok(())
}

fn main() -> std::io::Result<()> {
	// bench_grid()?;
	// try_serialization()?;
	// bench("results_new_gini.txt", false)?;
	// try_backup("./serialized/12.00000.serialized")?;

	// // Run full training on the setup with highest classification rate
	// bench("results_best.txt", &[0.69], &RandomForestBuilder {
	// 	n_trees: 100,
	// 	max_depth: 4,
	// 	bag_amount: 0.5,
	// }, true)?;

	let args = std::env::args().collect::<Vec<String>>();
	let action = args.get(1).expect("(1) Need to specify action (string)");
	let dataset_location = args.get(2).expect("(2) Need to specify dataset location (string)");
	let serializing_location = args.get(3).expect("(3) Need to specify serializing location (string)");
	
	match action.as_str() {
		"fit" => fit(dataset_location, serializing_location)?,
		"evaluate" => {
			let output_location = args.get(4).expect("(4) Need to specify output location (string)");
			evaluate(dataset_location, serializing_location, output_location)?;
		},
		_ => panic!("Action not recognized"),
	}

	Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialization_works() -> std::io::Result<()> {
        try_serialization()?;
        Ok(())
    }
}
