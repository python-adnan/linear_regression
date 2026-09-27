# Linear Regression from Scratch

A simple Python implementation of a simple **Linear Regression** model built from scratch using NumPy, Pandas, and Matplotlib. The model utilizes **Gradient Descent** to predict salaries based on years of experience.

## Features
- Custom cost function (Mean Squared Error) and gradient computations.
- Data normalization for faster gradient descent convergence.
- Visualizations matching the true data points against the linear model curve.

## Dataset
The script expects a file named `Salary_dataset.csv` inside the project root directory. It utilizes the following columns:
- `YearsExperience`: Predictor variable (X)
- `Salary`: Target variable (Y)

## Installation & Requirements

Ensure you have Python installed, then install the necessary dependencies:

```bash
pip install numpy matplotlib pandas
```

## How to Run

1. Place your `Salary_dataset.csv` file in the same directory as the script.
2. Execute the python script:
   ```bash
   python main.py
   ```
