# Only uncomment line 2 if you haven't yet installed the ISLP package. This will take a few minutes.
# !pip install ISLP

# Using the Credit Card dataset from ISLP
from ISLP import load_data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import warnings
warnings.filterwarnings('ignore')

df = load_data('Credit')

# dropping columns I won't use
df.drop(['ID','Ethnicity'],axis=1,inplace=True)

# better naming convention to the data for use
df.rename({
    'Income':'income',
    'Limit':'credit_limit',
    'Rating':'credit_rating',
    'Cards':'num_cards',
    'Age':'age',
    'Education':'level_of_education',
    'Gender':'gender',
    'Student':'is_student',
    'Married':'is_married',
    'Balance':'card_balance'
},axis=1,inplace=True)

# creating bool cols
df['gender'] = df['gender'].str.strip().str.lower().map({'male':0,'female':1})
df['is_student'] = df['is_student'].map({'No':0,'Yes':1})
df['is_married'] = df['is_married'].map({'No':0,'Yes':1})

# df.head()

# df.info()

# quick vizulations
fig, axes = plt.subplots(1, 2, figsize=(10,5))

# Boxplot for Income
sns.boxplot(y=df['income'], ax=axes[0], color='skyblue')
axes[0].set_title("Boxplot of Income")
axes[0].set_ylabel("Income")

# Boxplot for Card Balance
sns.boxplot(y=df['card_balance'], ax=axes[1], color='salmon')
axes[1].set_title("Boxplot of Card Balance")
axes[1].set_ylabel("Card Balance")

plt.tight_layout()
plt.show()

# income skews right
df['income'].plot(kind='hist', bins=20, title='income')
plt.gca().spines[['top', 'right',]].set_visible(False)

plt.figure(figsize=(8,5))
plt.scatter(df['income'],df['card_balance'],alpha=0.5)

plt.xlabel('Income')
plt.ylabel('Card Balance')
plt.title('Income vs Card Balance')

# plt.show()

plt.figure(figsize=(8,5))
plt.scatter(df['income'],df['card_balance'],alpha=0.5)

plt.xlabel('Income')
plt.ylabel('Card Balance')
plt.title('Income vs Card Balance')

plt.show()

# Make a copy to avoid modifying the original data
df_temp = df.copy()

# Replace 0 -> 'No' and 1 -> 'Yes'
df_temp['is_married'] = df_temp['is_married'].replace({0: 'No', 1: 'Yes'})
df_temp['is_student'] = df_temp['is_student'].replace({0: 'No', 1: 'Yes'})

# Melt the dataframe for Seaborn
df_melted = df_temp.melt(value_vars=['is_married', 'is_student'], var_name='Category', value_name='Response')

fig, ax = plt.subplots(figsize=(10, 5),facecolor='lightgray')

ax.grid(True,color='black',alpha=0.5)

fig.patch.set_facecolor('lightgray')

ax.set_facecolor('lightgray')

# Plot
sns.countplot(x='Category', hue='Response', data=df_melted)

# Labels and title
plt.xlabel("Category")
plt.ylabel("Count")
plt.title("Comparison of Marital Status and Student Status")
plt.legend(facecolor='darkgray')
plt.show()

df_temp['gender'] = df_temp['gender'].replace({0: 'Male', 1: 'Female'})

# Plot gender distribution
plt.figure(figsize=(6,4))
sns.countplot(x=df_temp['gender'], palette="Set2")  # Change color palette if needed

plt.xlabel("Gender")
plt.ylabel("Count")
plt.title("Distribution of Gender")

plt.show()

plt.figure(figsize=(8,5))
sns.lineplot(x=df['credit_rating'], y=df['credit_limit'], marker='o', linestyle='-')

# Labels and title
plt.xlabel("Credit Rating")
plt.ylabel("Credit Limit")
plt.title("Line Plot of Credit Limit vs. Credit Rating")

plt.show()

# Looking at feature correlation
corr_mtx = df.corr()

plt.figure(figsize=(10, 8))  # Set figure size
plt.gcf().set_facecolor('gray')
sns.heatmap(corr_mtx, annot=True, cmap='Greens', fmt=".2f", linewidths=0.5)

# Show the plot
plt.title("Credit Card Feature Correlation Heatmap")
plt.show()

# let's just grab a few of the more correlated X-variables
# Anything with an absolute value of 0.25 or greater
corr_mask = corr_mtx['card_balance'].abs() >= 0.25
features = corr_mtx['card_balance'][corr_mask]
features = features.reset_index()
features = features['index'].to_list()
features

data = df[features]
print('Shape of the data: ', data.shape)
data.head(3)

# Scaling the data so we can get a more understandable model
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

data[features] = scaler.fit_transform(data[features])

data.head(3)

# import libraries
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error

# set dependent/independent variables
y = data['card_balance']
X = data.drop('card_balance',axis=1)

# split data on a 0.25 threshold, meaning 1/4 data will be used for testing
# random state ensures "randomness" but also that we can create this bit of "randomness" (which seems antithetical to the idea of randomness, but I digress)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

# helper functions
# although mean_squared_error exists in the sklearn.metrics library,
# I wanted to take a more manual approach in scoring this
def get_mse(y_true, y_pred):
  """
  Manually computes the MSE of an OLS model
  """
  # get out y and y-hat values into vectors
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)

  squared_error = ((y_true - y_pred)**2)

  mean_square_error = np.mean(squared_error)

  return mean_square_error

# This performs manual Cross-Validation on an OLS model
def ols_cve(kf, X, y, scores_list):
  for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    X_train_sm = sm.add_constant(X_train)
    X_test_sm = sm.add_constant(X_test)

    model = sm.OLS(y_train, X_train_sm).fit()

    y_pred = model.predict(X_test_sm)

    scores_list.append(get_mse(y_test, y_pred))

    return scores_list


# Add constant (intercept) to X_train and X_test
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# Train OLS Model
model = sm.OLS(y_train, X_train_sm).fit()

# Make predictions on Train and Test sets
y_train_pred = model.predict(X_train_sm)
y_test_pred = model.predict(X_test_sm)

# Compute Train and Test MSE
train_mse = round(get_mse(y_train, y_train_pred),4)
test_mse = round(get_mse(y_test, y_test_pred),4)

print("Train MSE:", train_mse)
print("Test MSE:", test_mse)

# Compute Expected MSE
kf = KFold(n_splits=5, shuffle=True, random_state = 42) # what is randomness anyways?

mse_scores = []

expected_mse = round(np.mean(ols_cve(kf,X,y,scores_list=mse_scores)),4)

print("Expected MSE:", expected_mse)

# looking at the credit model from earlier:

# get y - \hat y from the model
residuals = model.resid

plt.figure(figsize=(8,5))
sns.scatterplot(x=y_train_pred, y=residuals, alpha=0.5)

plt.axhline(y=0,color='red',linestyle='--')

plt.xlabel("y_pred")
plt.ylabel("residuals")
plt.title('Checking for Homoscedasticity')
plt.show();

# Run OLS model from before
model = sm.OLS(y, X).fit()

# Get residuals
residuals = model.resid

# Compute correlation between residuals and predictors
corr_matrix = np.corrcoef(X.T, residuals)

corr_values = np.corrcoef(X.T, model.resid)[-1, :-1]
print("Correlation between predictors and residuals:", corr_values)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ensure X is a Pandas DataFrame
if isinstance(X, np.ndarray):
    X = pd.DataFrame(X, columns=[f"X_{i+1}" for i in range(X.shape[1])])  # Assign column names

# Get residuals from OLS model
residuals = model.resid  # OLS residuals

# Set up the subplots (2 rows, 2 columns)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Iterate through each predictor
for i, col in enumerate(X.columns[:4]):  # Ensure we only check X_1 to X_4
    sns.scatterplot(x=X[col], y=residuals, alpha=0.5, ax=axes[i])

    # Reference line at zero
    axes[i].axhline(y=0, color='red', linestyle='dashed')

    # Labels and title
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Residuals")
    axes[i].set_title(f"Residuals vs. {col}")

# Adjust layout
plt.tight_layout()
plt.show()

import numpy as np
import statsmodels.api as sm

# True parameters
beta_true = np.array([2, 3])  # Example: Intercept=2, Slope=3
num_simulations = 1000
n = 100  # Number of observations

# Store estimated betas
beta_estimates = np.zeros((num_simulations, len(beta_true)))

for i in range(num_simulations):
    # Generate random X and error
    X = np.column_stack((np.ones(n), np.random.rand(n)))  # Add intercept
    epsilon = np.random.normal(0, 1, size=n)  # Zero-mean error

    # Generate Y using true beta
    y = X @ beta_true + epsilon

    # Fit OLS model
    model = sm.OLS(y, X).fit()
    beta_estimates[i, :] = model.params  # Store estimated beta

# Compute mean of estimated betas
beta_mean = np.mean(beta_estimates, axis=0)
print("True beta:", beta_true)
print("Average estimated beta:", beta_mean)

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def gradient_descent_ols(X, y, alpha=0.01, num_iterations=1000, tol=1e-6):
    """
    Perform gradient descent to estimate OLS regression coefficients.

    Parameters:
        X (numpy array): Feature matrix (with intercept column included).
        y (numpy array): Target variable.
        alpha (float): Learning rate.
        num_iterations (int): Maximum number of iterations.
        tol (float): Convergence tolerance.

    Returns:
        beta (numpy array): Estimated coefficients.
        loss_history (list): Loss function values over iterations.
    """
    n, k = X.shape
    beta = np.zeros(k)
    loss_history = []

    for i in range(num_iterations):
        gradient = -2 * X.T @ (y - X @ beta) / n
        beta -= alpha * gradient  # Update beta

        # Compute current loss (Mean Squared Error)
        loss = np.mean((y - X @ beta) ** 2)
        loss_history.append(loss)

        # Check for convergence
        if i > 0 and abs(loss_history[-1] - loss_history[-2]) < tol:
            print(f"Converged at iteration {i}")
            break

    return beta, loss_history

# Generate synthetic data for testing
np.random.seed(42)
n, k = 100, 2  # 100 observations, 2 predictors
X = np.random.rand(n, k)
X = sm.add_constant(X)  # Add intercept column (X_0 = 1)
beta_true = np.array([2, 3, -1])  # True coefficients
y = X @ beta_true + np.random.randn(n)  # Generate y with noise

# Run Gradient Descent for OLS
beta_gd, loss_history = gradient_descent_ols(X, y, alpha=0.1, num_iterations=1000)

# Plot the Loss Function Over Iterations
plt.figure(figsize=(8,5))
plt.plot(loss_history, label="Loss (MSE)", color='blue')
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Gradient Descent Convergence for OLS")
plt.legend()
plt.grid()
plt.show()

# Print final estimated coefficients
print("Estimated Coefficients (Gradient Descent):", beta_gd)

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def gradient_descent_weight_decay(X, y, alpha=0.01, lambda_val=0.1, num_iterations=1000, tol=1e-6):
    """
    Perform gradient descent with weight decay (L2 regularization).

    Parameters:
        X (numpy array): Feature matrix (with intercept column included).
        y (numpy array): Target variable.
        alpha (float): Learning rate.
        lambda_val (float): Weight decay parameter.
        num_iterations (int): Number of gradient descent steps.
        tol (float): Convergence tolerance.

    Returns:
        beta (numpy array): Estimated coefficients.
        loss_history (list): Loss function values over iterations.
    """
    n, k = X.shape
    beta = np.zeros(k)
    loss_history = []  # Track loss over iterations

    for i in range(num_iterations):
        # Compute gradient with weight decay
        gradient = 2 * (X.T @ X @ beta - X.T @ y + lambda_val * beta) / n
        beta = (1 - 2 * alpha * lambda_val) * beta - alpha * gradient

        # Compute current loss (MSE with weight decay)
        loss = np.mean((y - X @ beta) ** 2) + lambda_val * np.sum(beta ** 2)
        loss_history.append(loss)

        # Check for convergence
        if i > 0 and abs(loss_history[-1] - loss_history[-2]) < tol:
            print(f"Converged at iteration {i}")
            break

    return beta, loss_history

# Generate synthetic data for testing
np.random.seed(42)
n, k = 100, 2  # 100 observations, 2 predictors
X = np.random.rand(n, k)
X = sm.add_constant(X)  # Add intercept column (X_0 = 1)
beta_true = np.array([2, 3, -1])  # True coefficients
y = X @ beta_true + np.random.randn(n)  # Generate y with noise

# Run Gradient Descent with Weight Decay
beta_wd_gd, loss_history_wd = gradient_descent_weight_decay(X, y, alpha=0.1, lambda_val=0.1, num_iterations=1000)

# Plot the Loss Function Over Iterations
plt.figure(figsize=(8,5))
plt.plot(loss_history_wd, label="Loss (MSE + Weight Decay)", color='purple')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Gradient Descent Convergence with Weight Decay")
plt.legend()
plt.grid()
plt.show()

# Print final estimated coefficients with weight decay
print("Estimated Coefficients (Gradient Descent with Weight Decay):", beta_wd_gd)

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def gradient_descent_ridge(X, y, alpha=0.01, lambda_val=1.0, num_iterations=1000, tol=1e-6):
    """
    Perform gradient descent to estimate Ridge Regression coefficients.

    Parameters:
        X (numpy array): Feature matrix (with intercept column included).
        y (numpy array): Target variable.
        alpha (float): Learning rate.
        lambda_val (float): Regularization parameter.
        num_iterations (int): Number of gradient descent steps.
        tol (float): Convergence tolerance.

    Returns:
        beta (numpy array): Estimated coefficients.
        loss_history (list): Loss function values over iterations.
    """
    n, k = X.shape  # Number of samples (n) and predictors (k)
    beta = np.zeros(k)  # Initialize beta with zeros
    loss_history = []  # Track loss over iterations

    for i in range(num_iterations):
        gradient = 2 * (X.T @ X @ beta + lambda_val * beta - X.T @ y) / n  # Compute Ridge gradient
        beta -= alpha * gradient  # Update beta

        # Compute current loss (Mean Squared Error + Ridge Penalty)
        loss = np.mean((y - X @ beta) ** 2) + lambda_val * np.sum(beta ** 2)
        loss_history.append(loss)

        # Check for convergence
        if i > 0 and abs(loss_history[-1] - loss_history[-2]) < tol:
            print(f"Converged at iteration {i}")
            break

    return beta, loss_history

# Generate synthetic data for testing
np.random.seed(42)
n, k = 100, 2  # 100 observations, 2 predictors
X = np.random.rand(n, k)
X = sm.add_constant(X)  # Add intercept column (X_0 = 1)
beta_true = np.array([2, 3, -1])  # True coefficients
y = X @ beta_true + np.random.randn(n)  # Generate y with noise

# Run Gradient Descent for Ridge Regression
beta_ridge_gd, loss_history_ridge = gradient_descent_ridge(X, y, alpha=0.1, lambda_val=1.0, num_iterations=1000)

# Plot the Loss Function Over Iterations for Ridge
plt.figure(figsize=(8,5))
plt.plot(loss_history_ridge, label="Loss (MSE + Ridge Penalty)", color='red')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Gradient Descent Convergence for Ridge Regression")
plt.legend()
plt.grid()
plt.show()

# Print final estimated coefficients for Ridge Regression
print("Estimated Coefficients (Ridge Gradient Descent):", beta_ridge_gd)


import numpy as np

def ridge_regression(X, y, lambda_val):
    """Computes Ridge Regression coefficients using closed-form solution."""
    n, k = X.shape
    I = np.eye(k)  # Identity matrix
    ridge_beta = np.linalg.inv(X.T @ X + lambda_val * I) @ X.T @ y
    return ridge_beta

# Example usage
lambda_val = 1.0  # Set lambda
ridge_beta_manual = ridge_regression(X, y, lambda_val)
print("Ridge Coefficients (Manual Calculation):", ridge_beta_manual)


import numpy as np

sample_num = 100
x_dim = 10
x = np.random.rand(sample_num, x_dim)
w_tar = np.random.rand(x_dim)
b_tar = np.random.rand(1)[0]
y = np.matmul(x, np.transpose([w_tar])) + b_tar
C = 1e-6

def ridge_regression_GD(x,y,C):
    x = np.insert(x,0,1,axis=1) # adding a feature 1 to x at beggining nxd+1
    x_len = len(x[0,:])
    w = np.zeros(x_len) # d+1
    t = 0
    eta = 3e-3
    summ = np.zeros(x_len)
    grad = np.zeros(x_len)
    losses = np.array([0])
    loss_stry = 0

    for i in range(50):
        for i in range(len(y)): # here we calculate the summation for all rows for loss and gradient
            summ = summ + (y[i,] - np.dot(w, x[i,])) * x[i,]
            loss_stry += (y[i,] - np.dot(w, x[i,]))**2

        losses = np.insert(losses, len(losses), loss_stry + C * np.dot(w, w))
        grad = -2 * summ + np.dot(2 * C,w)
        w -= eta * grad

        eta *= 0.9
        t += 1
        summ = np.zeros(1)
        loss_stry = 0

    return w[1:], w[0], losses

w, b, losses = ridge_regression_GD(x, y, C)
print("losses: ", losses)
print("b: ", b)
print("b_tar: ", b_tar)
print("w: ", w)
print("w_tar", w_tar)

x_pre = np.random.rand(3, x_dim)
y_tar = np.matmul(x_pre, np.transpose([w_tar])) + b_tar
y_pre = np.matmul(x_pre, np.transpose([w])) + b
print("y_pre: ", y_pre)

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz, cholesky

# Parameters
n = 100
p = 80

# Generate Data
Z = np.random.randn(n, p)
L = cholesky(toeplitz(0.6 ** np.arange(p)), lower=True)  # Add 'AR(1)' correlation
X = Z @ L
beta = np.random.uniform(2, 3, p)

def eye(n):
    return np.eye(n)

def calculate_ridge_mse(lambda_, nreps=1000):
    mse_values = []

    for _ in range(nreps):
        y = X @ beta + np.random.randn(n)
        beta_hat = np.linalg.solve(X.T @ X + lambda_ * eye(p), X.T @ y)
        mse_values.append(np.sum((beta - beta_hat) ** 2))

    return {"lambda": lambda_, "MSE": np.mean(mse_values)}

# Generate lambda grid
lambda_grid = np.logspace(-2, 2, 41)

# Compute MSE for Ridge Regression
ridge_mse = [calculate_ridge_mse(lmbda) for lmbda in lambda_grid]

# Convert results to NumPy array for easy plotting
lambda_values = np.array([entry["lambda"] for entry in ridge_mse])
mse_values = np.array([entry["MSE"] for entry in ridge_mse])

# Compute OLS MSE (lambda=0)
ols_mse = calculate_ridge_mse(0)["MSE"]

# Plot results
plt.figure(figsize=(8, 6))
plt.scatter(lambda_values, mse_values, label="Ridge MSE", color="blue")
plt.plot(lambda_values, mse_values, linestyle="solid", color="blue")
plt.axhline(y=ols_mse, color="red", linestyle="dashed", linewidth=2, label="OLS MSE")
plt.xscale("log")
plt.xlabel(r"$\lambda$")
plt.ylabel("Estimation Error (MSE)")
plt.title("Ridge Regularization Improves on OLS")
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Run Monte Carlo Simulation and generate results
num_simulations = 500
sample_sizes = [50, 100, 200]
beta_true = np.array([5, 2])  # Intercept = 5, Slope = 2

# Initialize storage for results
monte_carlo_results = {
    "sample_size": [],
    "bias_beta1": [],
    "variance_beta1": [],
    "in_sample_mse": [],
    "out_sample_mse": []
}

# Loop through different sample sizes
for n in sample_sizes:
    beta_estimates = np.zeros((num_simulations, len(beta_true)))
    in_sample_mse_list = []
    out_sample_mse_list = []

    for sim in range(num_simulations):
        # Generate synthetic dataset from linear DGP
        X = np.random.uniform(-3, 3, (n, 1))
        y = beta_true[0] + beta_true[1] * X.flatten() + np.random.normal(0, 1, n)  # Linear model with noise

        # Add intercept column
        X_with_intercept = sm.add_constant(X)

        # Fit OLS model
        model = sm.OLS(y, X_with_intercept).fit()
        beta_hat = model.params
        beta_estimates[sim, :] = beta_hat

        # In-sample MSE
        y_train_pred = model.predict(X_with_intercept)
        in_sample_mse = np.mean((y - y_train_pred) ** 2)
        in_sample_mse_list.append(in_sample_mse)

        # Generate test dataset for out-of-sample MSE
        X_test = np.random.uniform(-3, 3, (n, 1))
        y_test = beta_true[0] + beta_true[1] * X_test.flatten() + np.random.normal(0, 1, n)
        X_test_with_intercept = sm.add_constant(X_test)

        # Out-of-sample MSE
        y_test_pred = model.predict(X_test_with_intercept)
        out_sample_mse = np.mean((y_test - y_test_pred) ** 2)
        out_sample_mse_list.append(out_sample_mse)

    # Compute bias and variance for β1 (slope coefficient)
    beta1_mean = np.mean(beta_estimates[:, 1])  # Mean of estimated β1
    bias_beta1 = beta1_mean - beta_true[1]  # Bias = E[β_hat] - β
    variance_beta1 = np.var(beta_estimates[:, 1])  # Variance of β1 estimates

    # Store results
    monte_carlo_results["sample_size"].append(n)
    monte_carlo_results["bias_beta1"].append(bias_beta1)
    monte_carlo_results["variance_beta1"].append(variance_beta1)
    monte_carlo_results["in_sample_mse"].append(np.mean(in_sample_mse_list))
    monte_carlo_results["out_sample_mse"].append(np.mean(out_sample_mse_list))

# Extract results for plotting
sample_sizes = monte_carlo_results["sample_size"]
bias_beta1 = monte_carlo_results["bias_beta1"]
variance_beta1 = monte_carlo_results["variance_beta1"]
in_sample_mse = monte_carlo_results["in_sample_mse"]
out_sample_mse = monte_carlo_results["out_sample_mse"]

# Create figure for Bias, Variance, and MSE trends
plt.figure(figsize=(12, 8))

# Plot Bias of β1
plt.subplot(2, 2, 1)
plt.plot(sample_sizes, bias_beta1, marker='o', linestyle='-', color='red', label="Bias of β1")
plt.axhline(0, color='black', linestyle='dashed', label="Unbiased Line")
plt.xlabel("Sample Size (n)")
plt.ylabel("Bias of β1")
plt.title("Bias of β1 vs. Sample Size")
plt.legend()
plt.grid()

# Plot Variance of β1
plt.subplot(2, 2, 2)
plt.plot(sample_sizes, variance_beta1, marker='s', linestyle='-', color='blue', label="Variance of β1")
plt.xlabel("Sample Size (n)")
plt.ylabel("Variance of β1")
plt.title("Variance of β1 vs. Sample Size")
plt.legend()
plt.grid()

# Plot In-Sample MSE
plt.subplot(2, 2, 3)
plt.plot(sample_sizes, in_sample_mse, marker='o', linestyle='-', color='green', label="In-Sample MSE")
plt.xlabel("Sample Size (n)")
plt.ylabel("MSE")
plt.title("In-Sample MSE vs. Sample Size")
plt.legend()
plt.grid()

# Plot Out-of-Sample MSE
plt.subplot(2, 2, 4)
plt.plot(sample_sizes, out_sample_mse, marker='s', linestyle='-', color='purple', label="Out-of-Sample MSE")
plt.xlabel("Sample Size (n)")
plt.ylabel("MSE")
plt.title("Out-of-Sample MSE vs. Sample Size")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from ISLP import load_data  # Load ISLP dataset

# Load the Credit dataset from ISLP
credit_data = load_data('Credit')

# Convert categorical 'Student' variable to numeric (0 = No, 1 = Yes)
credit_data['Student'] = credit_data['Student'].map({'No': 0, 'Yes': 1})

# Drop missing values
credit_data = credit_data.dropna()

# Define features and target variable
features = ['Income', 'Limit', 'Rating', 'Student']  # Predictors
target = 'Balance'  # Response variable

def monte_carlo_ridge_credit(data, features, target, lambda_val=1.0, num_simulations=500, sample_sizes=[50, 100, 200]):
    """
    Perform Monte Carlo simulations on the Credit dataset using Ridge Regression
    to analyze bias, variance, and MSE.

    Parameters:
        data (DataFrame): The credit dataset.
        features (list): List of feature column names.
        target (str): Target variable.
        lambda_val (float): Regularization strength (Ridge penalty).
        num_simulations (int): Number of Monte Carlo simulations per sample size.
        sample_sizes (list): Different sample sizes to evaluate.

    Returns:
        results (dict): Dictionary containing bias, variance, and MSE metrics.
    """

    # Storage for results
    results = {
        "sample_size": [],
        "bias_beta1": [],
        "variance_beta1": [],
        "in_sample_mse": [],
        "out_sample_mse": []
    }

    # Convert dataset to NumPy arrays
    X_full = data[features].values  # Extract independent variables
    y_full = data[target].values  # Extract dependent variable

    # Add intercept column manually for Ridge Regression
    X_full = np.column_stack((np.ones(len(y_full)), X_full))

    # True coefficients (estimated from full dataset using Ridge Regression)
    model_full = Ridge(alpha=lambda_val, fit_intercept=False)
    model_full.fit(X_full, y_full)
    beta_true = model_full.coef_  # Treat full-sample Ridge estimates as "true" values

    for n in sample_sizes:
        beta_estimates = np.zeros((num_simulations, len(beta_true)))
        in_sample_mse_list = []
        out_sample_mse_list = []

        for _ in range(num_simulations):
            # Randomly sample training data
            sample_idx = np.random.choice(len(y_full), size=n, replace=False)
            X_train, y_train = X_full[sample_idx], y_full[sample_idx]

            # Fit Ridge Regression model
            ridge_model = Ridge(alpha=lambda_val, fit_intercept=False)
            ridge_model.fit(X_train, y_train)
            beta_hat = ridge_model.coef_
            beta_estimates[_, :] = beta_hat

            # Compute in-sample MSE
            y_train_pred = ridge_model.predict(X_train)
            in_sample_mse = np.mean((y_train - y_train_pred) ** 2)
            in_sample_mse_list.append(in_sample_mse)

            # Use remaining data as test set
            test_idx = np.setdiff1d(np.arange(len(y_full)), sample_idx)
            X_test, y_test = X_full[test_idx], y_full[test_idx]

            # Compute out-of-sample MSE
            y_test_pred = ridge_model.predict(X_test)
            out_sample_mse = np.mean((y_test - y_test_pred) ** 2)
            out_sample_mse_list.append(out_sample_mse)

        # Compute bias and variance for β1 (Income coefficient)
        beta1_mean = np.mean(beta_estimates[:, 1])  # Mean of estimated β1
        bias_beta1 = beta1_mean - beta_true[1]  # Bias = E[β_hat] - β
        variance_beta1 = np.var(beta_estimates[:, 1])  # Variance of β1 estimates

        # Store results
        results["sample_size"].append(n)
        results["bias_beta1"].append(bias_beta1)
        results["variance_beta1"].append(variance_beta1)
        results["in_sample_mse"].append(np.mean(in_sample_mse_list))
        results["out_sample_mse"].append(np.mean(out_sample_mse_list))

    # Convert results to NumPy arrays for plotting
    sample_sizes_np = np.array(results["sample_size"])
    bias_beta1_np = np.array(results["bias_beta1"])
    variance_beta1_np = np.array(results["variance_beta1"])
    in_sample_mse_np = np.array(results["in_sample_mse"])
    out_sample_mse_np = np.array(results["out_sample_mse"])

    # Plot Bias and Variance of β1 (Income) in Ridge Regression
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(sample_sizes_np, bias_beta1_np, marker='o', label="Bias of β1 (Income)")
    plt.axhline(0, color='r', linestyle='dashed', label="Unbiased Line")
    plt.xlabel("Sample Size (n)")
    plt.ylabel("Bias of β1")
    plt.title("Bias of β1 (Income) vs. Sample Size (Ridge)")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(sample_sizes_np, variance_beta1_np, marker='s', label="Variance of β1 (Income)")
    plt.xlabel("Sample Size (n)")
    plt.ylabel("Variance of β1")
    plt.title("Variance of β1 (Income) vs. Sample Size (Ridge)")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot In-Sample and Out-of-Sample MSE for Ridge Regression
    plt.figure(figsize=(8, 4))
    plt.plot(sample_sizes_np, in_sample_mse_np, marker='o', label="In-Sample MSE")
    plt.plot(sample_sizes_np, out_sample_mse_np, marker='s', label="Out-Sample MSE")
    plt.xlabel("Sample Size (n)")
    plt.ylabel("MSE")
    plt.title("In-Sample and Out-of-Sample MSE vs. Sample Size (Ridge)")
    plt.legend()
    plt.grid()
    plt.show()

    return results

# Run Monte Carlo Simulation for Ridge Regression on Credit Dataset
ridge_monte_carlo_results = monte_carlo_ridge_credit(
    credit_data,
    features=['Income', 'Limit', 'Rating', 'Student'],
    target='Balance',
    lambda_val=1.0  # Ridge penalty (adjustable)
)

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import Ridge

# Run Monte Carlo Simulation and generate results for Ridge Regression
num_simulations = 500
sample_sizes = [50, 100, 200]
beta_true = np.array([5, 2])  # Intercept = 5, Slope = 2
lambda_ridge = 1.0  # Regularization strength (λ)

ridge_monte_carlo_results = {
    "sample_size": [],
    "bias_beta1": [],
    "variance_beta1": [],
    "in_sample_mse": [],
    "out_sample_mse": []
}

for n in sample_sizes:
    beta_estimates = np.zeros((num_simulations, len(beta_true)))
    in_sample_mse_list = []
    out_sample_mse_list = []

    for sim in range(num_simulations):
        # Generate synthetic dataset
        X = np.random.uniform(-3, 3, (n, 1))
        y = beta_true[0] + beta_true[1] * X.flatten() + np.random.normal(0, 1, n)

        # Add intercept column
        X_with_intercept = sm.add_constant(X)

        # Fit Ridge Regression model
        ridge_model = Ridge(alpha=lambda_ridge, fit_intercept=False)
        ridge_model.fit(X_with_intercept, y)
        beta_hat = ridge_model.coef_  # Ridge estimated coefficients
        beta_estimates[sim, :] = beta_hat

        # Compute in-sample MSE
        y_train_pred = ridge_model.predict(X_with_intercept)
        in_sample_mse = np.mean((y - y_train_pred) ** 2)
        in_sample_mse_list.append(in_sample_mse)

        # Generate test dataset for out-of-sample MSE
        X_test = np.random.uniform(-3, 3, (n, 1))
        y_test = beta_true[0] + beta_true[1] * X_test.flatten() + np.random.normal(0, 1, n)
        X_test_with_intercept = sm.add_constant(X_test)

        # Compute out-of-sample MSE
        y_test_pred = ridge_model.predict(X_test_with_intercept)
        out_sample_mse = np.mean((y_test - y_test_pred) ** 2)
        out_sample_mse_list.append(out_sample_mse)

    # Compute bias and variance for β1 (slope coefficient)
    beta1_mean = np.mean(beta_estimates[:, 1])  # Mean of estimated β1
    bias_beta1 = beta1_mean - beta_true[1]  # Bias = E[β_hat] - β
    variance_beta1 = np.var(beta_estimates[:, 1])  # Variance of β1 estimates

    # Store results
    ridge_monte_carlo_results["sample_size"].append(n)
    ridge_monte_carlo_results["bias_beta1"].append(bias_beta1)
    ridge_monte_carlo_results["variance_beta1"].append(variance_beta1)
    ridge_monte_carlo_results["in_sample_mse"].append(np.mean(in_sample_mse_list))
    ridge_monte_carlo_results["out_sample_mse"].append(np.mean(out_sample_mse_list))

# Extract results for plotting
ridge_sample_sizes = ridge_monte_carlo_results["sample_size"]
ridge_bias_beta1 = ridge_monte_carlo_results["bias_beta1"]
ridge_variance_beta1 = ridge_monte_carlo_results["variance_beta1"]
ridge_in_sample_mse = ridge_monte_carlo_results["in_sample_mse"]
ridge_out_sample_mse = ridge_monte_carlo_results["out_sample_mse"]

# Create figure for Ridge Regression Bias, Variance, and MSE trends
plt.figure(figsize=(12, 8))

# Plot Bias of β1 for Ridge Regression
plt.subplot(2, 2, 1)
plt.plot(ridge_sample_sizes, ridge_bias_beta1, marker='o', linestyle='-', color='red', label="Bias of β1 (Ridge)")
plt.axhline(0, color='black', linestyle='dashed', label="Unbiased Line")
plt.xlabel("Sample Size (n)")
plt.ylabel("Bias of β1")
plt.title("Bias of β1 vs. Sample Size (Ridge)")
plt.legend()
plt.grid()

# Plot Variance of β1 for Ridge Regression
plt.subplot(2, 2, 2)
plt.plot(ridge_sample_sizes, ridge_variance_beta1, marker='s', linestyle='-', color='blue', label="Variance of β1 (Ridge)")
plt.xlabel("Sample Size (n)")
plt.ylabel("Variance of β1")
plt.title("Variance of β1 vs. Sample Size (Ridge)")
plt.legend()
plt.grid()

# Plot In-Sample MSE for Ridge Regression
plt.subplot(2, 2, 3)
plt.plot(ridge_sample_sizes, ridge_in_sample_mse, marker='o', linestyle='-', color='green', label="In-Sample MSE (Ridge)")
plt.xlabel("Sample Size (n)")
plt.ylabel("MSE")
plt.title("In-Sample MSE vs. Sample Size (Ridge)")
plt.legend()
plt.grid()

# Plot Out-of-Sample MSE for Ridge Regression
plt.subplot(2, 2, 4)
plt.plot(ridge_sample_sizes, ridge_out_sample_mse, marker='s', linestyle='-', color='purple', label="Out-of-Sample MSE (Ridge)")
plt.xlabel("Sample Size (n)")
plt.ylabel("MSE")
plt.title("Out-of-Sample MSE vs. Sample Size (Ridge)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Function to compute Ridge Regression MSE
def compute_ridge_mse(X, y, alpha=1.0):
    """
    Fits a Ridge Regression model and computes the Mean Squared Error (MSE).

    Parameters:
        X (numpy array): Feature matrix.
        y (numpy array): Target variable.
        alpha (float): Regularization strength for Ridge Regression.

    Returns:
        mse (float): Mean Squared Error of the Ridge Regression model.
    """
    X_linear = sm.add_constant(X)  # Add intercept for Ridge Regression
    ridge_model = Ridge(alpha=alpha, fit_intercept=False)  # Ridge model
    ridge_model.fit(X_linear, y)  # Fit Ridge model
    y_pred = ridge_model.predict(X_linear)  # Predictions
    mse = mean_squared_error(y, y_pred)  # Compute MSE
    return mse

# Define different sample sizes
sample_sizes = [50, 100, 500, 1000, 5000]
ridge_mse_results = []

for n in sample_sizes:
    X, y = generate_quadratic_dgp(n)  # Generate quadratic DGP data
    ridge_mse = compute_ridge_mse(X, y, alpha=1.0)  # Compute MSE with λ=1.0
    ridge_mse_results.append(ridge_mse)

# Plot Ridge MSE vs. Sample Size
plt.figure(figsize=(8, 5))
plt.plot(sample_sizes, ridge_mse_results, marker='o', linestyle='-', label="Ridge MSE (λ=1.0)", color='green')
plt.xlabel("Sample Size (n)")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Ridge Regression MSE in a Non-Linear Setting")
plt.legend()
plt.grid()
plt.show()

# Print Ridge MSE values for different sample sizes
for i, n in enumerate(sample_sizes):
    print(f"Sample Size {n}: Ridge MSE = {ridge_mse_results[i]:.4f}")

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Function to perform Ridge Regression with 5-Fold Cross-Validation to find optimal lambda
def ridge_regression_cv(X, y):
    """
    Performs Ridge Regression with 5-Fold Cross-Validation to determine the optimal regularization level.

    Parameters:
        X (numpy array): Feature matrix.
        y (numpy array): Target variable.

    Returns:
        best_ridge (Ridge model): Ridge model trained with the optimal lambda.
        best_lambda (float): Optimal regularization strength.
        best_mse (float): MSE using the optimal lambda.
    """
    X_linear = sm.add_constant(X)  # Add intercept for Ridge Regression
    ridge = Ridge()
    param_grid = {"alpha": np.logspace(-3, 3, 50)}  # Search for best lambda
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(ridge, param_grid, scoring='neg_mean_squared_error', cv=cv)
    grid_search.fit(X_linear, y)

    best_lambda = grid_search.best_params_["alpha"]
    best_ridge = Ridge(alpha=best_lambda).fit(X_linear, y)
    best_mse = -grid_search.best_score_  # Convert negative MSE to positive

    return best_ridge, best_lambda, best_mse

# Function to generate a non-linear quadratic dataset
def generate_quadratic_dgp(n, beta=2.0, noise_std=0.1):
    """
    Generates synthetic data based on a quadratic non-linear data-generating process (DGP):
        y = X^2 * beta + epsilon

    Parameters:
        n (int): Number of samples to generate.
        beta (float): True coefficient for the quadratic term.
        noise_std (float): Standard deviation of the noise term.

    Returns:
        X (numpy array): Feature values.
        y (numpy array): Target values.
    """
    np.random.seed(42)
    X = np.random.uniform(-3, 3, (n, 1))  # Generate X values in range [-3,3]
    y = beta * (X ** 2).flatten() + np.random.normal(0, noise_std, n)  # Quadratic relationship with noise
    return X, y

# Define sample sizes
sample_sizes = [50, 100, 500, 1000, 5000]
ridge_mse_results = []
ridge_optimal_lambda = []

# Compute Ridge MSE across different sample sizes and find optimal lambda
for n in sample_sizes:
    X, y = generate_quadratic_dgp(n)  # Generate quadratic DGP data
    best_ridge_model, best_lambda, best_mse = ridge_regression_cv(X, y)

    ridge_mse_results.append(best_mse)
    ridge_optimal_lambda.append(best_lambda)

# Plot Ridge MSE vs. Sample Size with Optimal Lambda
plt.figure(figsize=(8, 5))
plt.plot(sample_sizes, ridge_mse_results, marker='o', linestyle='-', label="Ridge MSE (Optimal λ)", color='green')
plt.xlabel("Sample Size (n)")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Optimal Ridge Regression MSE in a Non-Linear Setting")
plt.legend()
plt.grid()
plt.show()

# Print Ridge MSE and Optimal Lambda values for different sample sizes
for i, n in enumerate(sample_sizes):
    print(f"Sample Size {n}: Optimal λ = {ridge_optimal_lambda[i]:.4f}, Ridge MSE = {ridge_mse_results[i]:.4f}")

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import Ridge

# Define a new non-linear DGP
def generate_nonlinear_dgp(n, beta=np.array([5, 2]), noise_std=1.0):
    X = np.random.uniform(-3, 3, (n, 1))  # Generate X values in range [-3,3]
    y = beta[0] + beta[1] * (X.flatten() ** 2) + np.random.normal(0, noise_std, n)  # Quadratic relationship with noise
    return X, y

# Initialize storage for results
ridge_nonlinear_results = {
    "sample_size": [],
    "bias_beta1": [],
    "variance_beta1": [],
    "in_sample_mse": [],
    "out_sample_mse": []
}

lambda_ridge = 1.0  # Regularization strength (λ)
sample_sizes = [50, 100, 200]
num_simulations = 500

for n in sample_sizes:
    beta_estimates = np.zeros((num_simulations, 2))  # [Intercept, Quadratic term]
    in_sample_mse_list = []
    out_sample_mse_list = []

    for sim in range(num_simulations):
        # Generate synthetic dataset from non-linear DGP
        X, y = generate_nonlinear_dgp(n)

        # Transform X to include quadratic term (Polynomial Features)
        X_poly = np.column_stack((np.ones_like(X), X**2))  # Intercept and X^2

        # Fit Ridge Regression model
        ridge_model = Ridge(alpha=lambda_ridge, fit_intercept=False)
        ridge_model.fit(X_poly, y)
        beta_hat = ridge_model.coef_  # Ridge estimated coefficients
        beta_estimates[sim, :] = beta_hat

        # Compute in-sample MSE
        y_train_pred = ridge_model.predict(X_poly)
        in_sample_mse = np.mean((y - y_train_pred) ** 2)
        in_sample_mse_list.append(in_sample_mse)

        # Generate new test dataset for out-of-sample MSE
        X_test, y_test = generate_nonlinear_dgp(n)
        X_test_poly = np.column_stack((np.ones_like(X_test), X_test**2))  # Intercept and X^2

        # Compute out-of-sample MSE
        y_test_pred = ridge_model.predict(X_test_poly)
        out_sample_mse = np.mean((y_test - y_test_pred) ** 2)
        out_sample_mse_list.append(out_sample_mse)

    # Compute bias and variance for β1 (quadratic term)
    beta1_mean = np.mean(beta_estimates[:, 1])  # Mean of estimated β1
    bias_beta1 = beta1_mean - beta[1]  # Bias = E[β_hat] - β
    variance_beta1 = np.var(beta_estimates[:, 1])  # Variance of β1 estimates

    # Store results
    ridge_nonlinear_results["sample_size"].append(n)
    ridge_nonlinear_results["bias_beta1"].append(bias_beta1)
    ridge_nonlinear_results["variance_beta1"].append(variance_beta1)
    ridge_nonlinear_results["in_sample_mse"].append(np.mean(in_sample_mse_list))
    ridge_nonlinear_results["out_sample_mse"].append(np.mean(out_sample_mse_list))

# Extract results for plotting
ridge_sample_sizes = ridge_nonlinear_results["sample_size"]
ridge_bias_beta1 = ridge_nonlinear_results["bias_beta1"]
ridge_variance_beta1 = ridge_nonlinear_results["variance_beta1"]
ridge_in_sample_mse = ridge_nonlinear_results["in_sample_mse"]
ridge_out_sample_mse = ridge_nonlinear_results["out_sample_mse"]

# Create figure for Ridge Regression Bias, Variance, and MSE trends under a Non-Linear DGP
plt.figure(figsize=(12, 8))

# Plot Bias of β1 for Ridge Regression (Non-Linear)
plt.subplot(2, 2, 1)
plt.plot(ridge_sample_sizes, ridge_bias_beta1, marker='o', linestyle='-', color='red', label="Bias of β1 (Ridge)")
plt.axhline(0, color='black', linestyle='dashed', label="Unbiased Line")
plt.xlabel("Sample Size (n)")
plt.ylabel("Bias of β1")
plt.title("Bias of β1 vs. Sample Size (Ridge - Non-Linear)")
plt.legend()
plt.grid()

# Plot Variance of β1 for Ridge Regression (Non-Linear)
plt.subplot(2, 2, 2)
plt.plot(ridge_sample_sizes, ridge_variance_beta1, marker='s', linestyle='-', color='blue', label="Variance of β1 (Ridge)")
plt.xlabel("Sample Size (n)")
plt.ylabel("Variance of β1")
plt.title("Variance of β1 vs. Sample Size (Ridge - Non-Linear)")
plt.legend()
plt.grid()

# Plot In-Sample MSE for Ridge Regression (Non-Linear)
plt.subplot(2, 2, 3)
plt.plot(ridge_sample_sizes, ridge_in_sample_mse, marker='o', linestyle='-', color='green', label="In-Sample MSE (Ridge)")
plt.xlabel("Sample Size (n)")
plt.ylabel("MSE")
plt.title("In-Sample MSE vs. Sample Size (Ridge - Non-Linear)")
plt.legend()
plt.grid()

# Plot Out-of-Sample MSE for Ridge Regression (Non-Linear)
plt.subplot(2, 2, 4)
plt.plot(ridge_sample_sizes, ridge_out_sample_mse, marker='s', linestyle='-', color='purple', label="Out-of-Sample MSE (Ridge)")
plt.xlabel("Sample Size (n)")
plt.ylabel("MSE")
plt.title("Out-of-Sample MSE vs. Sample Size (Ridge - Non-Linear)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

