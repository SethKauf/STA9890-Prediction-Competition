if (!require("caret")) install.packages("caret")
if (!require("CVXR")) install.packages("CVXR")
if (!require("dplyr")) install.packages("dplyr")
if (!require("e1071")) install.packages("e1071")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("MASS")) install.packages("MASS")
if (!require("readr")) install.packages("readr")
if (!require("rpart")) install.packages("rpart")
if (!require("rpart.plot")) install.packages("rpart.plot")
if (!require("tibble")) install.packages("tibble")
install.packages("yardstick", repos = "https://cran.rstudio.com")

####################################################################
# LOAD DATA #
library(readr)



# load in encoded data
data <- read_csv('data/RR2/modeling_data_encoded.csv')

dim(data) # 48,140 x 89

library(dplyr)

glimpse(data)

# remove the secondary y-target
data <- data |>
  dplyr::select(-readmit_early)

y <- data$readmit_at_all
X <- data |> dplyr::select(-readmit_at_all)

set.seed(42)

# TRAIN TEST SPLIT #

# need to separate by patient_nbr so same patients don't end up in train and target
patient_ids <- unique(data$patient_nbr)
train_ids <- sample(patient_ids, size = 0.8 * length(patient_ids))

train_data <- data |> filter(patient_nbr %in% train_ids)
test_data <- data |> filter(!(patient_nbr %in% train_ids))

# drop patient_nbr
train_data <- train_data |> dplyr::select(-patient_nbr)
test_data <- test_data |> dplyr::select(-patient_nbr)

####################################################################
# NAIVE BAYES CLASSIFIER #

if (!require("yardstick")) install.packages("yardstick")

library(e1071) # NaiveBayes Classifier Library

nb_model <- naiveBayes(readmit_at_all ~ ., data = train_data)

# Predict Class
nb_pred <- predict(nb_model, test_data)

# Predict proba
nb_prob <- predict(nb_model, test_data, type="raw")[, 2]

library(caret)

# evaluate with Confusion Matrix
confusionMatrix(as.factor(nb_pred), as.factor(test_data$readmit_at_all))

####################################################################
# LDA CLASSIFIER #

library(MASS) # LDA Library

lda_model <- lda(readmit_at_all ~ ., data = train_data)

lda_pred <- predict(lda_model, test_data)

# classification report
confusionMatrix(as.factor(lda_pred$class), as.factor(test_data$readmit_at_all))

####################################################################
# SVM with CVX #
library(CVXR)

# X and y prep using pipe
X_train <- train_data |>
  dplyr::select(-readmit_at_all) |>
  as.matrix()

y_train <- ifelse(train_data$readmit_at_all == 1, 1, -1)

X_test <- test_data |>
  dplyr::select(-readmit_at_all) |>
  as.matrix()

y_test <- test_data$readmit_at_all

n <- nrow(X_train)
p <- ncol(X_train)

w <- Variable(p)
b <- Variable(1)
C <- 1  # You can tune this

# Hinge loss
hinge_loss <- sum(pos(1 - multiply(y_train, X_train %*% w + b)))

# Objective
objective <- Minimize(0.5 * sum_squares(w) + C * hinge_loss)
problem <- Problem(objective)

result <- solve(problem)

# Predicted margin values
margin_scores <- X_test %*% result$getValue(w) + result$getValue(b)

# Convert to class predictions
svm_pred <- ifelse(margin_scores >= 0, 1, 0)

confusionMatrix(as.factor(svm_pred), as.factor(y_test))

####################################################################
# LogReg with CVX #

# --- Shared Setup ---
n <- nrow(X_train)
p <- ncol(X_train)

# Define CVXR variables
beta <- Variable(p)
intercept <- Variable(1)
margin <- X_train %*% beta + intercept

# Prediction function
predict_logit <- function(result, beta, intercept, X) {
  b <- result$getValue(beta)
  i <- result$getValue(intercept)
  1 / (1 + exp(- (X %*% b + i)))
}

# =========================
# 1. Plain Logistic
# =========================
log_loss_plain <- sum(logistic(- y_train * margin))
problem_plain <- Problem(Minimize(log_loss_plain))
result_plain <- solve(problem_plain, solver = "ECOS")

prob_pred_plain <- predict_logit(result_plain, beta, intercept, X_test)
pred_class_plain <- ifelse(prob_pred_plain >= 0.5, 1, 0)
cm_plain <- confusionMatrix(as.factor(pred_class_plain), as.factor(y_test))
df_plain <- as.data.frame(cm_plain$table)
df_plain$model <- "Plain"

# =========================
# 2. Ridge Logistic
# =========================
lambda_ridge <- 1
log_loss_ridge <- sum(logistic(- y_train * margin)) + lambda_ridge * sum_squares(beta)
problem_ridge <- Problem(Minimize(log_loss_ridge))
result_ridge <- solve(problem_ridge, solver = "ECOS")

prob_pred_ridge <- predict_logit(result_ridge, beta, intercept, X_test)
pred_class_ridge <- ifelse(prob_pred_ridge >= 0.5, 1, 0)
cm_ridge <- confusionMatrix(as.factor(pred_class_ridge), as.factor(y_test))
df_ridge <- as.data.frame(cm_ridge$table)
df_ridge$model <- "Ridge"

# =========================
# 3. Lasso Logistic
# =========================
lambda_lasso <- 1
log_loss_lasso <- sum(logistic(- y_train * margin)) + lambda_lasso * norm1(beta)
problem_lasso <- Problem(Minimize(log_loss_lasso))
result_lasso <- solve(problem_lasso, solver = "ECOS")

prob_pred_lasso <- predict_logit(result_lasso, beta, intercept, X_test)
pred_class_lasso <- ifelse(prob_pred_lasso >= 0.5, 1, 0)
cm_lasso <- confusionMatrix(as.factor(pred_class_lasso), as.factor(y_test))
df_lasso <- as.data.frame(cm_lasso$table)
df_lasso$model <- "Lasso"

# Combine for plotting
df_all <- bind_rows(df_plain, df_ridge, df_lasso)

# Accuracy
cat("Plain Accuracy:", round(cm_plain$overall["Accuracy"], 4), "\n")
cat("Ridge Accuracy:", round(cm_ridge$overall["Accuracy"], 4), "\n")
cat("Lasso Accuracy:", round(cm_lasso$overall["Accuracy"], 4), "\n")

# Create individual plots
plot_cm <- function(df, title, fill_color) {
  ggplot(df, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), size = 5) +
    scale_fill_gradient(low = "white", high = fill_color) +
    labs(title = title, x = "Actual", y = "Predicted") +
    theme_minimal()
}

p1 <- plot_cm(df_plain, "Plain LogReg", "firebrick")
p2 <- plot_cm(df_ridge, "Ridge LogReg", "steelblue")
p3 <- plot_cm(df_lasso, "Lasso LogReg", "darkgreen")

library(patchwork)

# Combine plots
(p1 | p2 | p3) + plot_layout(guides = "collect") & theme(legend.position = "bottom")

####################################################################
# Decision Tree Classifier #

library(rpart)

# Make sure outcome is a factor for classification
train_data$readmit_at_all <- as.factor(train_data$readmit_at_all)
test_data$readmit_at_all  <- as.factor(test_data$readmit_at_all)

# Train single decision tree
tree_model <- rpart(
  readmit_at_all ~ ., 
  data = train_data,
  method = "class",
  control = rpart.control(
    minsplit = 10,        # allow smaller groups to split
    cp = 0.001,            # complexity parameter, lower = more splits
    maxdepth = 10          # allow deeper trees
  )
)

# Predict on test data
tree_preds <- predict(tree_model, newdata = test_data, type = "class")

# Evaluate
confusionMatrix(tree_preds, test_data$readmit_at_all)

# visualize
library(rpart.plot)

rpart.plot(tree_model, type = 3, extra = 101, fallen.leaves = TRUE)

####################################################################
# regular stack

## Convert SVM margin to probabilities
sigmoid <- function(x) {
  1 / (1 + exp(-x))
}
prob_pred_svm <- sigmoid(margin_scores)

# Tree probabilities
tree_probs <- predict(tree_model, test_data, type = "prob")
prob_pred_tree <- tree_probs[, "1"]

# Combine all model probabilities using correct variable names
ensemble_train <- data.frame(
  nb           = nb_prob,
  lda          = lda_pred$posterior[, 2],
  svm          = as.numeric(prob_pred_svm),
  tree         = as.numeric(prob_pred_tree),
  logreg_plain = as.numeric(prob_pred_plain),
  logreg_ridge = as.numeric(prob_pred_ridge),
  logreg_lasso = as.numeric(prob_pred_lasso)
)

# Normalize each prediction (min-max scaling)
ensemble_scaled <- ensemble_train %>%
  mutate(across(everything(), ~ (. - min(.)) / (max(.) - min(.))))

F_train <- as.matrix(ensemble_scaled)
n <- nrow(F_train)
k <- ncol(F_train)

# Response: convert 0/1 → -1/+1
y_train_ens <- ifelse(test_data$readmit_at_all == 1, 1, -1)

# CVXR optimization
W <- Variable(k)
b <- Variable(1)
margin <- F_train %*% W + b

log_loss <- sum(logistic(- y_train_ens * margin)) / n
lambda <- 0.0001
objective <- Minimize(log_loss + lambda * sum_squares(W))
problem <- Problem(objective)
result <- solve(problem)

# Predict and evaluate
weights <- result$getValue(W)
intercept <- result$getValue(b)

ensemble_scores <- F_train %*% weights + intercept
final_pred <- ifelse(ensemble_scores >= 0, 1, 0)

y_true <- as.numeric(as.character(test_data$readmit_at_all))
ensemble_cm <- confusionMatrix(as.factor(final_pred), as.factor(y_true))

cat("Ensemble Accuracy:", round(ensemble_cm$overall["Accuracy"], 4), "\n")
cat("Learned weights:\n")
print(round(weights, 4))

# Plot confusion matrix
ensemble_cm_df <- as.data.frame(ensemble_cm$table)

ggplot(data = ensemble_cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 5) +
  scale_fill_gradient(low = "white", high = "purple") +
  labs(
    title = "Stacked Ensemble Confusion Matrix",
    x = "Actual Label",
    y = "Predicted Label"
  ) +
  theme_minimal()

###########################################################################
# FairStacks
# 1. Combine model probabilities
# -------------------------
# --------------------------------------------------
# Use same normalized prediction matrix (F_train)
# --------------------------------------------------

# Response vector
y_train_ens <- ifelse(test_data$readmit_at_all == 1, 1, -1)

# --------------------------------------------------
# Estimate bias per model (Demographic Parity)
# --------------------------------------------------
# Define protected attribute, e.g., race_AfricanAmerican
protected_attr <- test_data$race_AfricanAmerican

# Compute demographic disparity per model
get_ddp <- function(preds, protected) {
  mean(preds[protected == 1]) - mean(preds[protected == 0])
}

bias_vec <- sapply(ensemble_scaled, get_ddp, protected = protected_attr)
bias_vec <- matrix(bias_vec, nrow = length(bias_vec), ncol = 1)

# --------------------------------------------------
# FairStacks Optimization
# --------------------------------------------------
n <- nrow(F_train)
k <- ncol(F_train)

W <- Variable(k)
b <- Variable(1)

margin <- F_train %*% W + b
log_loss <- sum(logistic(- y_train_ens * margin)) / n

lambda <- 0.0001  # regularization on weights
gamma <- 10       # fairness penalty

fair_penalty <- abs(t(bias_vec) %*% W)

objective <- Minimize(log_loss + lambda * sum_squares(W) + gamma * fair_penalty)
problem <- Problem(objective)
result <- solve(problem)

# --------------------------------------------------
# Predict and evaluate
# --------------------------------------------------
weights <- result$getValue(W)
intercept <- result$getValue(b)

ensemble_scores <- F_train %*% weights + intercept
final_pred <- ifelse(ensemble_scores >= 0, 1, 0)

y_true <- as.numeric(as.character(test_data$readmit_at_all))
ensemble_cm <- confusionMatrix(as.factor(final_pred), as.factor(y_true))

cat("FairStacks Accuracy:", round(ensemble_cm$overall["Accuracy"], 4), "\n")
cat("Learned FairStacks Weights:\n")
print(round(weights, 4))

# Plot confusion matrix
ensemble_cm_df <- as.data.frame(ensemble_cm$table)

ggplot(data = ensemble_cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 5) +
  scale_fill_gradient(low = "white", high = "purple") +
  labs(
    title = "FairStacks Ensemble Confusion Matrix",
    x = "Actual Label",
    y = "Predicted Label"
  ) +
  theme_minimal()

# Define model names — in same order as used in the ensemble
model_names <- c(
  "NaiveBayes", "LDA", "SVM", "DecisionTree",
  "LogReg_Plain", "LogReg_Ridge", "LogReg_Lasso"
)

# Extract weights from CVXR solution
fair_weights <- as.numeric(result$getValue(W))  # shape: [k x 1]

# Create data frame
weights_df <- tibble(
  Model = factor(model_names, levels = model_names),
  Weight = fair_weights
)

# Plot
ggplot(weights_df, aes(x = Model, y = Weight)) +
  geom_bar(stat = "identity", fill = "darkgreen") +
  theme_minimal() +
  labs(
    title = "FairStacks Ensemble Weights",
    x = NULL,
    y = "Model Weight"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))