# California Housing Price Prediction

An end-to-end machine learning project for predicting California house values using **Multiple Linear Regression**.

This project focuses on understanding the fundamentals of regression by implementing **linear regression and gradient descent from scratch using NumPy**, rather than relying entirely on a pre-built ML model.

## Project Overview

The goal is to predict median house values using demographic, geographic, and housing-related features from the California Housing dataset.

The project covers the complete basic machine learning workflow:

* Exploratory Data Analysis (EDA)
* Feature analysis and correlation
* Feature engineering
* Train/test splitting
* Feature normalization
* Multiple Linear Regression
* Gradient Descent from scratch
* Model evaluation using RMSE and R²
* Comparison with scikit-learn

## Dataset

The project uses the **California Housing dataset**.

The original dataset contains features such as:

* `MedInc` — Median income
* `HouseAge` — Median house age
* `AveRooms` — Average number of rooms
* `AveBedrms` — Average number of bedrooms
* `Population` — Block population
* `AveOccup` — Average number of household members
* `Latitude`
* `Longitude`

### Feature Engineering

An additional feature was created:

```python
df["roomperperson"] = df["AveRooms"] / df["AveOccup"]
```

This represents the approximate number of rooms available per person.

The addition of this feature improved the test RMSE from approximately:

```text
Without feature: ~0.74
With feature:    ~0.687
```

This demonstrates how feature engineering can improve the representation of information available to a linear model.

## Linear Regression From Scratch

Instead of directly relying on a machine learning library, the core optimization process was implemented using NumPy.

The model predicts:

```text
y_hat = XW + b
```

The parameters are updated using gradient descent:

```text
W = W - α * dW
b = b - α * db
```

where `α` is the learning rate.

The gradients are calculated using vectorized NumPy operations:

```python
dW = (1 / m) * (X.T @ error)
db = (1 / m) * np.sum(error)
```

This helped reinforce the underlying mathematics and mechanics of linear regression and gradient descent.

## Feature Normalization

Feature normalization was performed using statistics calculated from the training data:

```python
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train_norm = (X_train - mean) / std
X_test_norm = (X_test - mean) / std
```

The training mean and standard deviation are reused for the test set to avoid data leakage.

## Model Evaluation

Two evaluation metrics were used.

### RMSE

Root Mean Squared Error measures the typical magnitude of prediction errors.

```text
RMSE ≈ 0.687
```

Lower RMSE indicates better predictive performance.

### R² Score

R² measures the proportion of variance in the target that is explained by the model.

```text
R² ≈ 0.635
```

This means the linear model explains approximately **63.5% of the variation** in house values on the test set.

## Results

| Metric | Result |
| ------ | -----: |
| RMSE   | ~0.687 |
| R²     | ~0.635 |

The results demonstrate that a linear model can capture a substantial portion of the relationship between the available features and house values, while also showing its limitations when relationships are nonlinear.

For example, geographic features such as latitude and longitude can have complex relationships with house prices that are difficult for a simple linear model to capture.


## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

## Project Structure

```text
california-housing-regression/
│
├── California_Housing_Regression.ipynb
├── README.md
└── requirements.txt
```

## Future Improvements

this project can be extended by experimenting with:

* Polynomial Regression
* Ridge Regression
* Lasso Regression
* Regularization
* Additional feature engineering
* More advanced regression models
* Comparison of different models using the same evaluation metrics

The goal is to use this project as a baseline and understand how different machine learning techniques improve upon simple linear regression.