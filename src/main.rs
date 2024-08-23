use ruscikit::{Dataset, LinearRegression, mean_squared_error, r2_score, train_test_split, StandardScaler};
use ruscikit::model::Estimator;
use ndarray::{arr1, arr2};

fn main() {
    println!("Welcome to ruscikit!");

    // Create a larger dataset
    let features = arr2(&[
        [1.0, 1.0], [1.0, 2.0], [2.0, 2.0], [2.0, 3.0],
        [3.0, 3.0], [3.0, 4.0], [4.0, 4.0], [4.0, 5.0],
        [5.0, 5.0], [5.0, 6.0], [6.0, 6.0], [6.0, 7.0],
    ]);
    let targets = arr1(&[
        6.0, 8.0, 9.0, 11.0, 
        12.0, 14.0, 15.0, 17.0, 
        18.0, 20.0, 21.0, 23.0
    ]);

    // Create a Dataset
    let mut dataset = Dataset::new(features, Some(targets));

    // Apply StandardScaler
    dataset.apply_standard_scaler().unwrap();
    println!("Applied StandardScaler to the dataset");

    // Split the data
    let (train_data, test_data) = train_test_split(&dataset, 0.2, Some(42)).unwrap();
    
    println!("Training set size: {}", train_data.n_samples());
    println!("Test set size: {}", test_data.n_samples());

    // Create and fit the model
    let mut model = LinearRegression::new();
    model.fit(&train_data).unwrap();

    // Make predictions on test data
    let predictions = model.predict(&test_data.features().to_owned()).unwrap();
    
    // Calculate metrics
    if let Some(targets) = test_data.targets() {
        let mse = mean_squared_error(&targets.to_owned(), &predictions);
        let r2 = r2_score(&targets.to_owned(), &predictions);
        
        println!("Mean Squared Error on test data: {:.4}", mse);
        println!("R-squared Score on test data: {:.4}", r2);
    }

    // Make predictions on new data
    let new_data = arr2(&[[7.0, 7.0], [8.0, 8.0]]);
    let mut scaler = StandardScaler::new();
    scaler.fit(dataset.features());
    let scaled_new_data = scaler.transform(new_data.view()).unwrap();
    
    match model.predict(&scaled_new_data) {
        Ok(predictions) => println!("Predictions for new data: {:?}", predictions),
        Err(e) => println!("Error making predictions: {}", e),
    }
}
