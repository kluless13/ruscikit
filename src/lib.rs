//! Ruscikit: A Rust implementation of scikit-learn functionality

pub mod dataset;
pub mod model;
pub mod preprocessing;
pub mod linear_models;
pub mod metrics;
pub mod model_selection;

pub use dataset::{Dataset, train_test_split};
pub use model::ModelParameters;
pub use linear_models::LinearRegression;
pub use metrics::{mean_squared_error, r2_score};
pub use model_selection::{KFold, cross_val_score};
pub use preprocessing::StandardScaler;
