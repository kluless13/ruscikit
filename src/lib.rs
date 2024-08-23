//! Ruscikit: A Rust implementation of scikit-learn functionality

pub mod dataset;
pub mod model;
pub mod preprocessing;
pub mod linear_models;

pub use dataset::Dataset;
pub use model::ModelParameters;
pub use linear_models::LinearRegression;
