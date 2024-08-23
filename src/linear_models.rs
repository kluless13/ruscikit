use ndarray::{Array1, Array2, s};
use ndarray_linalg::Solve;

use crate::Dataset;
use crate::model::{Estimator, ModelParameters};

#[derive(Clone)]
pub struct LinearRegression {
    coefficients: Option<Array1<f64>>,
    intercept: Option<f64>,
    params: ModelParameters,
}

impl LinearRegression {
    pub fn new() -> Self {
        LinearRegression {
            coefficients: None,
            intercept: None,
            params: ModelParameters::new(),
        }
    }

    pub fn coefficients(&self) -> Option<&Array1<f64>> {
        self.coefficients.as_ref()
    }

    pub fn intercept(&self) -> Option<f64> {
        self.intercept
    }
}

impl Estimator for LinearRegression {
    fn fit(&mut self, dataset: &Dataset) -> Result<(), String> {
        let x = dataset.features();
        let y = dataset.targets().ok_or("No target values provided.")?;

        let mut x_with_bias = Array2::ones((x.nrows(), x.ncols() + 1));
        x_with_bias.slice_mut(s![.., 1..]).assign(&x);

        let coeffs = x_with_bias.t().dot(&x_with_bias)
            .solve(&x_with_bias.t().dot(&y))
            .map_err(|e| format!("Failed to solve linear system: {}", e))?;

        self.intercept = Some(coeffs[0]);
        self.coefficients = Some(coeffs.slice(s![1..]).to_owned());

        Ok(())
    }

    fn predict(&self, features: &Array2<f64>) -> Result<Array1<f64>, String> {
        match (&self.coefficients, &self.intercept) {
            (Some(coef), Some(intercept)) => {
                Ok(features.dot(coef) + *intercept)
            },
            _ => Err("Model has not been fitted.".to_string()),
        }
    }

    fn get_params(&self) -> &ModelParameters {
        &self.params
    }

    fn set_params(&mut self, params: ModelParameters) -> Result<(), String> {
        self.params = params;
        Ok(())
    }
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}
