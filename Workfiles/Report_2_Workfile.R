library(caret)
library(CVXR)
library(dplyr)
library(DT)
library(ggplot2)
library(ggrepel)
library(glmnet)
library(glue)
library(gridExtra)
library(ISLR)
library(lubridate)
library(Metrics)
library(patchwork)
library(plotly)
library(purrr)
library(psych)
library(readr)
library(reshape2)
library(scales)
library(stringr)
library(tidyverse)

# if (!require("CVXR")) install.packages("CVXR")

set.seed(100)  # For reproducibility
n <- 400       # Set up a moderately high-dimensional problem
p <- 1000
s <- 5
sigma <- 2

# Lasso works best for IID Gaussian data
X <- matrix(rnorm(n * p), 
            nrow=n, ncol=p)

# 'True' coefficients are mainly sparse with 5 non-zero values
beta_star <- matrix(rep(0, p), ncol=1)
beta_star[1:s] <- 3

# Generate observed response from OLS DGP
y <- X %*% beta_star + rnorm(n, sd=sigma)

## We are now ready to apply CVXR
####  Also see discussion at
####  https://www.cvxpy.org/examples/machine_learning/lasso_regression.html

beta <- Variable(p) # Create 'beta' as a CVX _variable_ to be optimized

# Per theory, about the right level of regularization to be used here
lambda  <- sigma * sqrt(s * log(p) / n) 
loss    <- 1/(2 * n) * sum((y - X %*% beta)^2) # MSE Loss
penalty <- lambda * sum(abs(beta))

objective <- Problem(Minimize(loss + penalty))
beta_hat  <- solve(objective)$getValue(beta)

plot(beta_hat, 
     xlab="Coefficient Number", 
     ylab="Lasso Estimate", 
     main="CVX Lasso Solution", 
     col="red4", 
     pch=16)


getwd()

df <- read_csv('data/RR2/diabetic_data.csv')
ids <- read_csv('data/RR2/IDS_mapping.csv')


glimpse(df)
ids