use ndarray::{Array1, Array2, ArrayView2};
use crate::Dataset;

#[derive(Clone, Debug)]
pub struct StandardScaler {
    mean: Option<Array1<f64>>,
    std: Option<Array1<f64>>,
}

impl StandardScaler {
    pub fn new() -> Self {
        StandardScaler {
            mean: None,
            std: None,
        }
    }

    pub fn fit(&mut self, x: ArrayView2<f64>) {
        let mean = x.mean_axis(ndarray::Axis(0)).unwrap();
        let std = x.std_axis(ndarray::Axis(0), 0.);
        
        self.mean = Some(mean);
        self.std = Some(std);
    }

    pub fn transform(&self, x: ArrayView2<f64>) -> Result<Array2<f64>, String> {
        if self.mean.is_none() || self.std.is_none() {
            return Err("StandardScaler has not been fitted.".to_string());
        }

        let mean = self.mean.as_ref().unwrap();
        let std = self.std.as_ref().unwrap();

        Ok((&x - mean) / std)
    }

    pub fn fit_transform(&mut self, x: ArrayView2<f64>) -> Array2<f64> {
        self.fit(x);
        self.transform(x).unwrap()
    }
}

impl Dataset {
    pub fn apply_standard_scaler(&mut self) -> Result<(), String> {
        let mut scaler = StandardScaler::new();
        let scaled_features = scaler.fit_transform(self.features());
        self.set_features(scaled_features);
        Ok(())
    }
}
