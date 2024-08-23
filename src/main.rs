use ruscikit::{Dataset, LinearRegression};
use ruscikit::model::Estimator;
use ndarray::{arr1, arr2};

fn main() {
    println!("Welcome to ruscikit!");

    // Create a dataset
    let features = arr2(&[[1.0, 1.0], [1.0, 2.0], [2.0, 2.0], [2.0, 3.0]]);
    let targets = arr1(&[6.0, 8.0, 9.0, 11.0]);
    let dataset = Dataset::new(features, Some(targets));
    
    println!("Dataset created with {} samples", dataset.features().nrows());

    // Create and fit the model
    let mut model = LinearRegression::new();
    match model.fit(&dataset) {
        Ok(_) => println!("Model fitted successfully"),
        Err(e) => println!("Error fitting model: {}", e),
    }

    // Make predictions
    let new_data = arr2(&[[3.0, 5.0], [4.0, 4.0]]);
    match model.predict(&new_data) {
        Ok(predictions) => println!("Predictions: {:?}", predictions),
        Err(e) => println!("Error making predictions: {}", e),
    }
}
