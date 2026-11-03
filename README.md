# House Price Prediction Using Multiple Linear Regression

## Project Overview

This project implements a **Multiple Linear Regression** machine learning model to predict house prices based on various property features.

Unlike Simple Linear Regression, which uses only one independent variable, Multiple Linear Regression considers multiple features such as house area, number of bedrooms, bathrooms, stories, parking spaces, and furnishing status to estimate house prices.

The project was developed using Python in Google Colab as a practical exercise to understand the complete machine learning workflow.

## Project Objectives

- Understand and implement Multiple Linear Regression.
- Explore and preprocess a real-world housing dataset.
- Perform Exploratory Data Analysis (EDA).
- Identify relationships between housing features and prices.
- Train a regression model using Scikit-learn.
- Evaluate model performance using regression evaluation metrics.
- Generate house price predictions using multiple independent variables.

## Dataset

**Dataset:** Housing Price Prediction Dataset

**Source:** [Kaggle – Housing Price Prediction](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)

Dataset information:
- Total records: 545
- Original columns: 13
- Independent variables: 12
- Dependent variable: 1 (`price`)

### Dataset Features

| Feature | Description |
|---|---|
| price | House selling price (target variable) |
| area | House area |
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| stories | Number of stories |
| mainroad | Whether the house is near a main road |
| guestroom | Whether a guestroom is available |
| basement | Whether a basement is available |
| hotwaterheating | Availability of hot water heating |
| airconditioning | Availability of air conditioning |
| parking | Number of parking spaces |
| prefarea | Whether the house is in a preferred area |
| furnishingstatus | Furnishing condition of the house |

## Technologies and Libraries

- **Python** – Programming language
- **Pandas** – Data manipulation and analysis
- **NumPy** – Numerical operations
- **Matplotlib** – Data visualization
- **Seaborn** – Statistical data visualization
- **Scikit-learn** – Machine learning and model evaluation
- **Google Colab** – Development environment

## Project Workflow

### 1. Data Loading and Understanding
- Imported the required Python libraries.
- Loaded the housing dataset using Pandas.
- Examined the first five records.
- Inspected dataset dimensions, data types, and descriptive statistics.

### 2. Data Cleaning and Preprocessing
- Checked for missing values.
- Checked for duplicate records.
- Inspected categorical variables and their unique values.
- Identified potential outliers using boxplots.
- Converted binary categorical variables into numerical values:
  - Yes = 1
  - No = 0
- Applied One-Hot Encoding to the `furnishingstatus` column.
- Used `drop_first=True` to avoid redundant dummy variables.

### 3. Exploratory Data Analysis (EDA)

Performed exploratory analysis using:
- Descriptive statistics.
- House price distribution histogram.
- Area vs. house price scatterplot.
- Correlation analysis.
- Correlation heatmap.

**Key observations:**
- House prices show a positively skewed distribution.
- House area has a positive correlation with house prices (0.54).
- Bathrooms have a positive correlation with house prices (0.52).
- Air conditioning has a positive correlation with house prices (0.45).
- Unfurnished status has a negative correlation with house prices (-0.28).

### 4. Feature Selection and Data Splitting

Separated the dataset into independent and dependent variables.

- Independent variables (X): 13 features.
- Dependent variable (y): `price`.

Split the dataset into training and testing sets using an 80:20 ratio.

| Dataset | Records |
|---|---:|
| Training | 436 |
| Testing | 109 |

Used `random_state=42` to ensure reproducibility.

### 5. Model Building and Training

Implemented Multiple Linear Regression using Scikit-learn.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
```

Generated predictions using:

```python
y_pred = model.predict(X_test)
```

Also examined the model intercept and regression coefficients to understand the contribution of individual features.

### 6. Model Evaluation

Evaluated model performance using the following metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

## Model Performance

The initial Multiple Linear Regression model achieved the following results on the test dataset:

| Evaluation Metric | Result |
|---|---:|
| MAE | 970,043.40 |
| MSE | 1,754,319,000,000 |
| RMSE | 1,324,507 |
| R² Score | 0.6529 (65.29%) |

### Performance Interpretation

- **MAE:** The average absolute prediction error is approximately 970,043 price units.
- **MSE:** Measures the average squared prediction error.
- **RMSE:** Indicates a prediction error scale of approximately 1.32 million price units, with greater sensitivity to large errors.
- **R² Score:** The model explains approximately 65.29% of the variation in house prices in the test dataset.

These results represent the initial model performance and provide a baseline for potential improvements.

## Visualizations

The project includes the following visualizations:

1. Distribution of House Prices
2. House Area vs. House Price
3. Correlation Heatmap
4. Boxplots of Numerical Features

Additional model evaluation visualizations can include:
- Actual vs. Predicted Prices
- Residual Plot
- Distribution of Residuals

## Future Improvements

Potential improvements to explore:

- Perform further regression diagnostic checks.
- Investigate multicollinearity using Variance Inflation Factor (VIF).
- Examine the influence of outliers.
- Compare training and testing performance.
- Experiment with feature selection.
- Compare Multiple Linear Regression with Ridge and Lasso Regression.
- Explore feature transformations to improve model performance.

## Project Structure

```text
House-Price-Prediction/
│
├── Housing.csv
├── House_Price_Prediction.ipynb
└── README.md
```

## How to Run the Project

1. Clone this repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Navigate to the project directory.

3. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

4. Open the Jupyter Notebook or upload it to Google Colab.

5. Ensure `Housing.csv` is available in the working directory.

6. Run the notebook cells sequentially.

## Learning Outcomes

Through this project, I gained practical experience in:

- Understanding Multiple Linear Regression.
- Working with numerical and categorical data.
- Performing data cleaning and preprocessing.
- Applying binary encoding and One-Hot Encoding.
- Conducting Exploratory Data Analysis.
- Understanding correlation and regression coefficients.
- Splitting datasets into training and testing sets.
- Training a machine learning regression model.
- Evaluating regression performance using different metrics.

## Author

**Chalani Fernando**

BSc (Hons) Computer Science

---

*This project was developed for learning and practical exploration of supervised machine learning using Multiple Linear Regression.*
