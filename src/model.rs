use crate::Dataset;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

pub trait Estimator: Clone {
    fn fit(&mut self, dataset: &Dataset) -> Result<(), String>;
    fn predict(&self, features: &Array2<f64>) -> Result<Array1<f64>, String>;
    fn get_params(&self) -> &ModelParameters;
    fn set_params(&mut self, params: ModelParameters) -> Result<(), String>;
}

#[derive(Clone, Debug)]
pub enum ParamValue {
    Float(f64),
    Int(i64),
    Bool(bool),
    String(String),
}

#[derive(Clone, Debug)]
pub struct ModelParameters {
    params: HashMap<String, ParamValue>,
}

impl ModelParameters {
    pub fn new() -> Self {
        ModelParameters {
            params: HashMap::new(),
        }
    }

    pub fn set(&mut self, key: &str, value: ParamValue) {
        self.params.insert(key.to_string(), value);
    }

    pub fn get(&self, key: &str) -> Option<&ParamValue> {
        self.params.get(key)
    }
}

impl Default for ModelParameters {
    fn default() -> Self {
        Self::new()
    }
}
