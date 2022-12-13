use std::io::{Read, Write};
use std::time::{Instant, Duration};

use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::dataset::Dataset;
use crate::classifier::Classifier;
use crate::decision_tree::{DecisionTreeBuilder, DecisionTree};
use crate::functions::{most_frequent};

struct Foo {
    i: usize,
    time: Duration,
    durations: Vec<Duration>,
    whole: Instant
}

pub struct RandomForestBuilder {
    pub n_trees: usize,
    pub max_depth: usize,
    pub bag_amount: f64,
}

impl Default for RandomForestBuilder {
    fn default() -> Self {
        Self {
            n_trees: 100,
            max_depth: 32,
            bag_amount: 0.5,
        }
    }
}

impl RandomForestBuilder {
    pub fn fit(&self, dataset: Dataset) -> RandomForestClassifier {
        let feature_len = (dataset.features_len() as f64).sqrt().ceil() as usize;

        let foo = Foo {
            i: 0,
            time: Duration::from_secs(0),
            durations: Vec::new(),
            whole: Instant::now()
        };

        let mutex = std::sync::Mutex::new(foo);
        let arc = std::sync::Arc::new(mutex);

        let forest = self.get_rngs()
            .enumerate()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(_i, mut rng)| {
                let now = Instant::now();
                let tree = self.fit_tree(&mut rng, &dataset, feature_len);
                let elapsed_time = now.elapsed();

                let mut guard = arc.lock().unwrap();
                guard.i += 1;
                guard.time += elapsed_time;
                guard.durations.push(elapsed_time);

                let progress = guard.i as f64 / self.n_trees as f64;
                let total = guard.whole.elapsed().as_millis() as f64 / progress;
                let estimated = total - guard.whole.elapsed().as_millis() as f64 * (1.0 / progress as f64 - 1.0);

                println!(
                    "[{:.1}%] {:.2} m left ({:.2} m)",
                    progress * 100.0,
                    (total - estimated) / 1000.0 / 60.0,
                    elapsed_time.as_secs_f64() / 60.0,
                );

                tree
            })
            .collect::<Vec<_>>();

        let guard = arc.lock().unwrap();
        println!("-------\n[{:.2} m] Done {:.2} m cpu time", guard.whole.elapsed().as_secs_f64() / 60.0, guard.time.as_secs_f64() / 60.0);

        RandomForestClassifier {
            forest
        }
    }

    fn fit_tree<R: Rng + ?Sized>(&self, rng: &mut R, dataset: &Dataset, feature_len: usize) -> DecisionTree {
        let builder = DecisionTreeBuilder {
            max_features: Some(feature_len),
            max_depth: self.max_depth,
        };

        let max_samples = (dataset.rows_len() as f64 * self.bag_amount) as usize;
        let bootstrapped = dataset.bootstrap(rng, max_samples);

        builder.fit(rng, bootstrapped)
    }

    fn get_rngs(&self) -> impl Iterator<Item = StdRng> {
        let seed_u64: u64 = rand::thread_rng().gen();
        let mut seed = [0u8; 32];
        (&mut seed[0..8]).copy_from_slice(&seed_u64.to_be_bytes()[..]);
        let mut rng = StdRng::from_seed(seed);
        (0..self.n_trees).map(move |_| {
            let mut seed = [0u8; 32];
            rng.fill(&mut seed);
            StdRng::from_seed(seed)
        })
    }
}

pub struct RandomForestClassifier {
    forest: Vec<DecisionTree>,
}

impl Classifier for RandomForestClassifier {
    fn predict(&self, x: &[f64]) -> f64 {
        most_frequent(self.forest.iter().map(|v| v.predict(x)))
    }

    fn serialize<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_u16::<BigEndian>(self.forest.len() as u16)?;

        for tree in &self.forest {
            tree.serialize(writer)?;
        }

        Ok(())
    }

	fn deserialize<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let len = reader.read_u16::<BigEndian>()?;

        let forest = (0..len)
            .map(|_| DecisionTree::deserialize(reader))
            .collect::<std::io::Result<Vec<DecisionTree>>>()?;

        Ok(Self {
            forest
        })
    }
}