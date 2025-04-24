# Research Report Prediction Competition
# Starting Code from Project homepage https://michael-weylandt.com/STA9890/competition.html
library(readr)
library(dplyr)

TRAINING_SAMPLES <- read_csv("http://michael-weylandt.com/STA9890/competition_data/assessment_history_train.csv.gz")
TEST_POINTS <- read_csv("http://michael-weylandt.com/STA9890/competition_data/assessment_history_test.csv.gz")

AVERAGE_ASSESSMENT <- mean(TRAINING_SAMPLES$TARGET)

TEST_PREDICTIONS <- TEST_POINTS |> 
  select(acct) |>
  mutate(TARGET = AVERAGE_ASSESSMENT) |>
  rename(ACCOUNT = acct)

PATH = 'predictions/'

write_csv(TEST_PREDICTIONS, paste(PATH,"kaggle_sub_r.csv"))