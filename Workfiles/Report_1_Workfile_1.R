if (!require("caret")) install.packages("caret")
if (!require("dplyr")) install.packages("dplyr")
if (!require("DT")) install.packages("DT")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("ggrepel")) install.packages("ggrepel")
if (!require("glue")) install.packages("glue")
if (!require("gridExtra")) install.packages("gridExtra")
if (!require("ISLR")) install.packages("ISLR")
if (!require("lubridate")) install.packages("lubridate")
if (!require("Metrics")) install.packages("Metrics")
if (!require("patchwork")) install.packages("patchwork")
if (!require("plotly")) install.packages("plotly")
if (!require("purrr")) install.packages("purrr")
if (!require("psych")) install.packages("psych")
if (!require("readr")) install.packages("readr")
if (!require("reshape2")) install.packages("reshape2")
if (!require("scales")) install.packages("scales")
if (!require("stringr")) install.packages("stringr")
if (!require("tidyverse")) install.packages("tidyverse")

library(caret)
library(dplyr)
library(DT)
library(ggplot2)
library(ggrepel)
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


# Load the Credit dataset
df <- Credit

# Drop columns: ID and Ethnicity
df <- df |> select(-ID, -Ethnicity)

# Rename columns for better naming convention
df <- df |>
  rename(
    income = Income,
    credit_limit = Limit,
    credit_rating = Rating,
    num_cards = Cards,
    age = Age,
    level_of_education = Education,
    gender = Gender,
    is_student = Student,
    is_married = Married,
    card_balance = Balance
  )

# Create boolean columns
df <- df |>
  mutate(
    gender = ifelse(str_trim(tolower(gender)) == "male", 0, 1),
    is_student = ifelse(is_student == "Yes", 1, 0),
    is_married = ifelse(is_married == "Yes", 1, 0)
  )

head(df)
glimpse(df)

############################## PLOTS ##############################

# Boxplots
p1 <- ggplot(df, aes(y = income)) +
  geom_boxplot(fill = "skyblue") +
  labs(title = "Boxplot of Income", y = "Income")

p2 <- ggplot(df, aes(y = card_balance)) +
  geom_boxplot(fill = "salmon") +
  labs(title = "Boxplot of Card Balance", y = "Card Balance")

# Show side-by-side
# library(gridExtra)
grid.arrange(p1, p2, ncol = 2)

ggplot(df, aes(x = income)) +
  geom_histogram(bins = 20, fill = "steelblue", color = "white") +
  labs(title = "Income", x = "Income", y = "Count") +
  theme_minimal()

ggplot(df, aes(x = income, y = card_balance)) +
  geom_point(alpha = 0.5) +
  labs(
    title = "Income vs Card Balance",
    x = "Income",
    y = "Card Balance"
  ) +
  theme_minimal()

ggplot(df, aes(x = income, y = card_balance)) +
  geom_point(alpha = 0.5) +
  labs(
    title = "Income vs Card Balance",
    x = "Income",
    y = "Card Balance"
  ) +
  theme_minimal()

df_temp <- df |>
  mutate(
    is_married = ifelse(is_married == 1, "Yes", "No"),
    is_student = ifelse(is_student == 1, "Yes", "No"),
    gender = ifelse(gender == 1, "Female", "Male")
  )

df_melted <- df_temp |>
  select(is_married, is_student) |>
  pivot_longer(cols = everything(), names_to = "Category", values_to = "Response")

ggplot(df_melted, aes(x = Category, fill = Response)) +
  geom_bar(position = "dodge") +
  labs(
    title = "Comparison of Marital Status and Student Status",
    x = "Category", y = "Count"
  ) +
  theme_minimal() +
  theme(panel.background = element_rect(fill = "lightgray"),
        plot.background = element_rect(fill = "lightgray"),
        legend.background = element_rect(fill = "darkgray"))

ggplot(df_temp, aes(x = gender)) +
  geom_bar(fill = "mediumseagreen") +
  labs(title = "Distribution of Gender", x = "Gender", y = "Count") +
  theme_minimal()

ggplot(df, aes(x = credit_rating, y = credit_limit)) +
  geom_line() +
  geom_point() +
  labs(
    title = "Line Plot of Credit Limit vs. Credit Rating",
    x = "Credit Rating",
    y = "Credit Limit"
  ) +
  theme_minimal()

# Calculate correlation matrix
corr_mtx <- df |>
  select(where(is.numeric)) |>
  cor()

# Plot heatmap
melted_corr <- melt(corr_mtx)

ggplot(melted_corr, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(value, 2))) +
  scale_fill_gradient(low = "white", high = "darkgreen") +
  theme_minimal() +
  labs(
    title = "Credit Card Feature Correlation Heatmap",
    x = "", y = ""
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.background = element_rect(fill = "gray"))

# Grab features with |correlation| >= 0.25
selected_features <- corr_mtx[,"card_balance"] %>%
  abs() %>%
  `>=`(0.25) %>%
  which() %>%
  names()

data <- df |> select(all_of(selected_features))

cat("Shape of the data: ", dim(data)[1], "rows and", dim(data)[2], "columns\n")
head(data, 3)

# Scale numeric features
scaled_data <- df |>
  select(all_of(selected_features)) |>
  mutate(across(-card_balance, scale))  # scale everything except target

head(scaled_data, 3)

# library(caret)

set.seed(42)
train_index <- createDataPartition(scaled_data$card_balance, p = 0.75, list = FALSE)

train_data <- scaled_data[train_index, ]
test_data <- scaled_data[-train_index, ]

# Separate X and y
X_train <- train_data %>% select(-card_balance)
y_train <- train_data$card_balance

X_test <- test_data %>% select(-card_balance)
y_test <- test_data$card_balance

# Add intercept manually (base R lm includes it by default)
model <- lm(card_balance ~ ., data = train_data)

# Predict
y_train_pred <- predict(model, newdata = X_train)
y_test_pred <- predict(model, newdata = X_test)

# library(Metrics)

get_mse <- function(y_true, y_pred) {
  mean((y_true - y_pred)^2)
}

train_mse <- round(get_mse(y_train, y_train_pred), 4)
test_mse <- round(get_mse(y_test, y_test_pred), 4)

cat("Train MSE:", train_mse, "\n")
cat("Test MSE:", test_mse, "\n")

set.seed(42)
kf <- createFolds(scaled_data$card_balance, k = 5, returnTrain = TRUE)

ols_cve <- function(kf, X, y) {
  scores <- c()
  
  for (fold in kf) {
    X_train_fold <- X[fold, ]
    y_train_fold <- y[fold]
    
    X_test_fold <- X[-fold, ]
    y_test_fold <- y[-fold]
    
    train_fold_df <- cbind(card_balance = y_train_fold, X_train_fold)
    test_fold_df <- cbind(card_balance = y_test_fold, X_test_fold)
    
    model <- lm(card_balance ~ ., data = train_fold_df)
    y_pred <- predict(model, newdata = X_test_fold)
    
    scores <- c(scores, get_mse(y_test_fold, y_pred))
  }
  
  return(scores)
}

X <- scaled_data %>% select(-card_balance)
y <- scaled_data$card_balance

cv_scores <- ols_cve(kf, X, y)
expected_mse <- round(mean(cv_scores), 4)
cat("Expected MSE:", expected_mse, "\n")

residuals <- model$residuals

ggplot(data = NULL, aes(x = y_train_pred, y = residuals)) +
  geom_point(color = "steelblue", alpha = 0.75) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Checking for Homoscedasticity",
    x = "y_pred",
    y = "Residuals"
  ) +
  theme_minimal()


# Fit model on full data
final_model <- lm(card_balance ~ ., data = scaled_data)

# Residuals
residuals <- final_model$residuals

# Correlation between residuals and predictors
resid_corr <- cor(scaled_data %>% select(-card_balance), residuals)

cat("Correlation between predictors and residuals:\n")
print(round(resid_corr, 4))

# library(patchwork)
# library(purrr)

# Get residuals from the model
residuals <- final_model$residuals

# original variable names and X index
original_names <- selected_features[selected_features != "card_balance"]
new_colnames <- paste0("X_", seq_along(original_names))
colnames_map <- setNames(original_names, new_colnames)

# rename X columns to X_1, X_2, ...
X <- scaled_data %>%
  select(all_of(original_names)) %>%
  setNames(names(colnames_map))

# Combine residuals
X_resid <- X %>%
  mutate(residuals = residuals)

# Create plots with expression-based labels
plots <- map2(
  names(colnames_map)[1:4],
  colnames_map[1:4],
  ~ {
    xi_index <- as.numeric(sub("X_", "", .x))  # extract number
    varname <- .y
    
    ggplot(X_resid, aes_string(x = .x, y = "residuals")) +
      geom_point(color = "steelblue", alpha = 0.7) +
      geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
      labs(
        title = bquote("residuals vs." ~ .(varname) ~ "(" ~ X[.(xi_index)] ~ ")"),
        x = bquote(.(varname) ~ "(" ~ X[.(xi_index)] ~ ")"),
        y = "residuals"
      ) +
      theme_minimal()
  }
)

wrap_plots(plots, nrow = 1)


## Simulate Gradient Descent

# Function
gradient_descent_weight_decay <- function(X, y, alpha = 0.01, lambda_val = 0.1, num_iterations = 1000, tol = 1e-6) {
  # Add intercept column if not already present
  if (!all(X[,1] == 1)) {
    X <- cbind(1, X)
  }
  
  n <- nrow(X)
  k <- ncol(X)
  beta <- rep(0, k)
  loss_history <- numeric()
  
  for (i in 1:num_iterations) {
    # Compute gradient with weight decay
    gradient <- 2 * (t(X) %*% X %*% beta - t(X) %*% y + lambda_val * beta) / n
    beta <- (1 - 2 * alpha * lambda_val) * beta - alpha * gradient
    
    # Compute loss (MSE + L2 penalty)
    loss <- mean((y - X %*% beta)^2) + lambda_val * sum(beta^2)
    loss_history[i] <- loss
    
    # Convergence check
    if (i > 1 && abs(loss_history[i] - loss_history[i - 1]) < tol) {
      message("Converged at iteration ", i)
      break
    }
  }
  
  list(beta = beta, loss_history = loss_history)
}

# Set simulation params
set.seed(42)
n <- 100
k <- 2
X <- matrix(runif(n * k), nrow = n)
X <- cbind(1, X)  # Add intercept

beta_true <- c(2, 3, -1)
y <- X %*% beta_true + rnorm(n)

# Gradient Descent with Weight Decay
result <- gradient_descent_weight_decay(X, y, alpha = 0.1, lambda_val = 0.1, num_iterations = 1000)

beta_wd_gd <- result$beta
loss_history_wd <- result$loss_history

cat("Estimated Coefficients (Gradient Descent with Weight Decay):\n")
print(round(beta_wd_gd, 4))


df_loss <- data.frame(
  Iteration = seq_along(loss_history_wd),
  Loss = loss_history_wd
)

# --- 1. Find convergence point
loss_diff <- abs(diff(df_loss$Loss))
converged_at <- which(loss_diff < 1e-6)[1] + 1  # +1 because diff() shortens length

# --- 2. Find bend point (max second derivative)
first_deriv <- diff(df_loss$Loss)
second_deriv <- diff(first_deriv)
bend_at <- which.max(abs(second_deriv)) + 2  # +2 to align with original index

# Points for annotation
converge_point <- df_loss[converged_at, ]
bend_point <- df_loss[bend_at, ]

# Calculate offset positions for label placement
converge_point <- converge_point %>%
  mutate(label_x = Iteration - 1,
         label_y = Loss + 0.25)

bend_point <- bend_point %>%
  mutate(label_x = Iteration + 15,
         label_y = Loss)

# Build the plot
p <- ggplot(df_loss, aes(x = Iteration, y = Loss)) +
  geom_line(color = "purple", linewidth = 1) +
  
  # 🔴 Convergence point and callout
  geom_segment(data = converge_point, aes(x = Iteration, y = Loss, xend = label_x, yend = label_y),
               color = "red", linetype = "-") +
  geom_point(data = converge_point, aes(x = Iteration, y = Loss),
             color = "red", size = 3) +
  geom_text(data = converge_point,
            aes(x = label_x, y = label_y,
                label = paste0("Converged at:\nIteration ", Iteration)),
            color = "red", fontface = "bold", size = 4,
            hjust = 0.5) +
  
  # 🟠 Bend point and callout
  geom_segment(data = bend_point, aes(x = Iteration, y = Loss, xend = label_x, yend = label_y),
               color = "orange", linetype = "-") +
  geom_point(data = bend_point, aes(x = Iteration, y = Loss),
             color = "orange", size = 3) +
  geom_text(data = bend_point,
            aes(x = label_x, y = label_y,
                label = paste0("Natural Stopping Point:\nIteration ", Iteration)),
            color = "orange", fontface = "bold", size = 4,
            hjust = 0) +
  
  labs(
    title = "Gradient Descent Convergence with Weight Decay",
    x = "Iterations",
    y = "Loss (MSE + L2 Penalty)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(hjust = 0.5),
    plot.background = element_rect(fill = "gray95", color = NA),
    panel.background = element_rect(fill = "gray90", color = "gray70"),
    panel.border = element_rect(color = "gray40", fill = NA),
    panel.grid.major = element_line(color = "gray70", linetype = "dotted"),
    panel.grid.minor = element_blank()
  )

# Make interactive
ggplotly(p)

# plot loss
df_loss <- data.frame(
  Iteration = seq_along(loss_history_wd),
  Loss = loss_history_wd
)

ggplot(df_loss, aes(x = Iteration, y = Loss)) +
  geom_line(color = "purple", linewidth = 1) +
  labs(
    title = "Gradient Descent Convergence with Weight Decay",
    x = "Iterations",
    y = "Loss (MSE + L2 Penalty)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(hjust = 0.5),
    plot.background = element_rect(fill = "gray95", color = NA),
    panel.background = element_rect(fill = "gray90", color = "gray70"),
    panel.border = element_rect(color = "gray40", fill = NA),
    panel.grid.major = element_line(color = "gray70", linetype = "dotted"),
    panel.grid.minor = element_blank()
  )
