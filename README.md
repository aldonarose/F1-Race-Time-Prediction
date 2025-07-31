# F1 Race Time Prediction

This project utilizes machine learning to predict the average lap times of Formula 1 drivers in a race, based on their qualifying performance. By training a model on historical data, it forecasts the likely race pace of drivers, which is a key indicator of the potential race winner.
## Table of Contents
* [Project Overview](#project-overview)
* [The fastf1 Library](#the-fastf1-library)
* [The Model: Gradient Boosting Regressor](#the-model-gradient-boosting-regressor)
* [Output and Interpretation](#output-and-interpretation)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
## Project Overview
This repository contains a Jupyter Notebook (F1RaceTimePredictions.ipynb) that demonstrates a complete machine learning workflow for predicting F1 race outcomes. The central concept is to use qualifying times as a primary feature to predict a driver's average lap time during a race. A faster predicted lap time suggests a higher chance of winning.

The project follows these key steps:

1. **Data Collection**: Lap time data from the 2024 Australian Grand Prix is loaded using the fastf1 library.
  
2. **Data Preprocessing**: The data is cleaned by removing incomplete lap records and converting lap times into a numerical format (total seconds).

3. **Feature Engineering**: A hypothetical qualifying dataset for a future "2025 GP" is created to serve as the input for our prediction model.

4. **Model Training**: A Gradient Boosting Regressor is trained on the 2024 race data to learn the relationship between qualifying times and race lap times.

5. **Prediction**: The trained model predicts the average race lap times for the drivers in the "2025 GP" dataset.

6. **Evaluation**: The model's accuracy is evaluated using the Mean Absolute Error (MAE) metric on a held-out test set.
## The fastf1 Library
A cornerstone of this project is the fastf1 library, a powerful open-source Python tool designed specifically for accessing and analyzing Formula 1 data. It provides an easy-to-use interface to a vast amount of F1 data, including:
* Live timing data
* Historical race data
* Car telemetry (speed, throttle, brake, etc.)
* Tyre information
* Weather data
* Session results

For this project, fastf1 is essential for programmatically fetching detailed lap-by-lap data from a specific race weekend, which forms the basis of our model's training set. The library integrates seamlessly with popular data analysis libraries like pandas, making data manipulation and analysis straightforward.

## The Model: Gradient Boosting Regressor
A GradientBoostingRegressor from the scikit-learn library was chosen for this prediction task. This is an ensemble learning technique that builds a strong predictive model by sequentially combining multiple "weak" models, typically decision trees.

Here's why it's a suitable choice for this project:

* **High Accuracy**: Gradient Boosting is known for its high predictive accuracy and often outperforms other algorithms, especially with well-structured, tabular data.

* **Handles Complex Relationships**: It can capture complex, non-linear relationships between variables, such as the intricate connection between a driver's one-lap qualifying pace and their sustained performance over a full race distance.

* **Robustness**: The iterative nature of the algorithm, where each new tree corrects the errors of the previous ones, makes the model robust and less sensitive to outliers in the data.
## Output and Interpretation
The final output of the notebook is a ranked list of drivers from the hypothetical "2025 GP" based on their predicted average race lap time. The driver with the lowest predicted time is forecasted as the most likely race winner.
```
Predicted 2025 GP Winner

             Driver  Predicted_RaceTime (s)
6   Charles Leclerc               83.208047
0      Lando Norris               83.251367
9      Carlos Sainz               83.468523
2   Max, Verstappen               83.810061
```

The notebook also calculates the Mean Absolute Error (MAE) of the model, which was found to be approximately 3.87 seconds. This value represents the average absolute difference between the model's predicted lap times and the actual lap times in the test set. In this context, it tells us that, on average, the model's predictions are off by about 3.87 seconds, giving us a clear understanding of its accuracy.
## Getting Started
To run this project on your local machine, follow the steps below.
### Prerequisites
You will need to have Python 3 installed.

The project depends on the following Python libraries:
* `pandas`
* `numpy`
* `scikit-learn`
* `fastf1`
### Installation
You can install all the required libraries using pip:
```
pip install pandas numpy scikit-learn fastf1
```
## Usage
The entire project is contained within the F1RaceTimePredictions.ipynb Jupyter Notebook. To run the analysis and see the predictions:

1. Launch Jupyter Notebook or JupyterLab.
2. Open the F1RaceTimePredictions.ipynb file.
3. Run the cells in sequential order.

This will execute the data loading, preprocessing, model training, and prediction steps, ultimately displaying the predicted race results.
