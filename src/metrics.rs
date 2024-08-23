use ndarray::Array1;

pub fn mean_squared_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let diff = y_true - y_pred;
    diff.mapv(|x| x.powi(2)).mean().unwrap_or(f64::INFINITY)
}

pub fn r2_score(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
    let ss_tot: f64 = y_true.mapv(|x| (x - y_true.mean().unwrap_or(0.0)).powi(2)).sum();
    let ss_res: f64 = y_true.iter().zip(y_pred.iter())
        .map(|(&yt, &yp)| (yt - yp).powi(2))
        .sum();
    1.0 - (ss_res / ss_tot)
}
