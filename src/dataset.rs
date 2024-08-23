use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Debug)]
pub struct Dataset {
    features: Array2<f64>,
    targets: Option<Array1<f64>>,
    feature_names: Option<Vec<String>>,
    target_name: Option<String>,
}

impl Dataset {
    pub fn new(features: Array2<f64>, targets: Option<Array1<f64>>) -> Self {
        Dataset {
            features,
            targets,
            feature_names: None,
            target_name: None,
        }
    }

    pub fn features(&self) -> ArrayView2<f64> {
        self.features.view()
    }

    pub fn targets(&self) -> Option<ArrayView1<f64>> {
        self.targets.as_ref().map(|t| t.view())
    }

    pub fn set_feature_names(&mut self, names: Vec<String>) -> Result<(), String> {
        if names.len() != self.features.ncols() {
            return Err(format!(
                "Number of feature names ({}) must match number of features ({})",
                names.len(),
                self.features.ncols()
            ));
        }
        self.feature_names = Some(names);
        Ok(())
    }

    pub fn set_target_name(&mut self, name: String) {
        self.target_name = Some(name);
    }

    pub fn get_feature_names(&self) -> Option<&Vec<String>> {
        self.feature_names.as_ref()
    }

    pub fn get_target_name(&self) -> Option<&String> {
        self.target_name.as_ref()
    }

    pub fn n_samples(&self) -> usize {
        self.features.nrows()
    }

    pub fn n_features(&self) -> usize {
        self.features.ncols()
    }

    pub fn set_features(&mut self, features: Array2<f64>) {
        self.features = features;
    }

    pub fn select_samples(&self, indices: &[usize]) -> Dataset {
        let features = self.features.select(ndarray::Axis(0), indices);
        let targets = self.targets.as_ref().map(|t| t.select(ndarray::Axis(0), indices));
        
        Dataset {
            features,
            targets,
            feature_names: self.feature_names.clone(),
            target_name: self.target_name.clone(),
        }
    }
}

pub fn train_test_split(
    dataset: &Dataset,
    test_size: f64,
    random_seed: Option<u64>,
) -> Result<(Dataset, Dataset), String> {
    if test_size <= 0.0 || test_size >= 1.0 {
        return Err("test_size must be between 0 and 1".to_string());
    }

    let n_samples = dataset.n_samples();
    let n_test = (n_samples as f64 * test_size).round() as usize;
    let n_train = n_samples - n_test;

    let mut rng = ChaCha8Rng::seed_from_u64(random_seed.unwrap_or(42));
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng);

    let train_indices = &indices[0..n_train];
    let test_indices = &indices[n_train..];

    Ok((
        dataset.select_samples(train_indices),
        dataset.select_samples(test_indices),
    ))
}
