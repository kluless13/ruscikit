use ndarray::Array1;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use crate::Dataset;
use crate::model::Estimator;

pub struct KFold {
    n_splits: usize,
    shuffle: bool,
    random_seed: Option<u64>,
}

impl KFold {
    pub fn new(n_splits: usize, shuffle: bool, random_seed: Option<u64>) -> Self {
        KFold {
            n_splits,
            shuffle,
            random_seed,
        }
    }

    pub fn split(&self, dataset: &Dataset) -> Vec<(Vec<usize>, Vec<usize>)> {
        let n_samples = dataset.n_samples();
        let mut indices: Vec<usize> = (0..n_samples).collect();
        
        if self.shuffle {
            let mut rng = ChaCha8Rng::seed_from_u64(self.random_seed.unwrap_or(42));
            indices.shuffle(&mut rng);
        }

        let fold_size = n_samples / self.n_splits;
        let mut folds = Vec::with_capacity(self.n_splits);

        for i in 0..self.n_splits {
            let test_start = i * fold_size;
            let test_end = if i == self.n_splits - 1 { n_samples } else { (i + 1) * fold_size };

            let test_indices = indices[test_start..test_end].to_vec();
            let train_indices = indices[0..test_start].iter()
                .chain(indices[test_end..].iter())
                .cloned()
                .collect();

            folds.push((train_indices, test_indices));
        }

        folds
    }
}

pub fn cross_val_score<E: Estimator>(
    estimator: &E,
    dataset: &Dataset,
    cv: &KFold,
    scoring: fn(&Array1<f64>, &Array1<f64>) -> f64,
) -> Result<Vec<f64>, String> {
    let folds = cv.split(dataset);
    let mut scores = Vec::with_capacity(cv.n_splits);

    for (train_indices, test_indices) in folds {
        let train_data = dataset.select_samples(&train_indices);
        let test_data = dataset.select_samples(&test_indices);

        let mut model = estimator.clone();
        model.fit(&train_data)?;

        let predictions = model.predict(&test_data.features().to_owned())?;
        let true_values = test_data.targets().ok_or("No target values found")?;

        let score = scoring(&true_values.to_owned(), &predictions);
        scores.push(score);
    }

    Ok(scores)
}
