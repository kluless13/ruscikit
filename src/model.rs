use std::collections::HashMap;
use crate::Dataset;

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

pub trait Estimator {
    fn fit(&mut self, dataset: &Dataset) -> Result<(), String>;
    fn get_params(&self) -> &ModelParameters;
    fn set_params(&mut self, params: ModelParameters) -> Result<(), String>;
}
