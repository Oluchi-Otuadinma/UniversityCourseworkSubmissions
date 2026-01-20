---
title: "Stats_coursework"
author: "Oluchi_Otuadinma"
date: "2024-11-22"
output: pdf_document
---
install.packages('MASS')
install.packages('ISLR2')
install.packages('caret')
install.packages('dplyr')
install.packages('FNN')

library(MASS)
library(ISLR2)
library(dplyr)
library(caret)
library(ggplot2)
library(FNN)
library(nnet)


setwd("E:/RStudio/Projects") 
Boston <- read.csv("boston_subset.cvs") 
head(Boston)

# produced the boston dataset to aid in understanding
summary(Boston)
head(Boston)

# (a)
# For reproducability
set.seed(35)

#seq_len creates a sequence of integers from 1 to n and assigns it to the rows in the boston dataset using nrow. 
#The sample function then randomly selects out of those numbers from til it reaches 66% of the total.
train_indices <- sample(seq_len(nrow(Boston)), size = 0.66 * nrow(Boston))

train_data <- Boston[train_indices, ] 
test_data <- Boston[-train_indices, ] 


# (b)
# Use the cor function to find correlation over 0.5 and remove it from the data
head(train_data)                                   

cor_matrix <- cor(train_data[, !(colnames(train_data) %in% 'crim')])                      
cor_matrix
columns_to_remove <- findCorrelation(
  cor_matrix,
  cutoff = 0.5,
  verbose = FALSE,
  names = TRUE,
)
columns_to_remove
reduced_data <- train_data %>% select(-nox, -indus, -lstat, -tax, -dis, -age, -medv)

low_corr <- lm(crim ~ ., reduced_data)
summary(low_corr)
plot(reduced_data)

# (c)
# I can reject the null hypothesis, H0 : βj = 0, for rm and rad as their p-values are less than 0.05 and are therefore significant.

# (d)
lm.fit <- lm(crim ~ zn + chas + rm + rad + ptratio, data = reduced_data)
summary(lm.fit)
lm.fit <- lm(crim ~ chas + rm + rad + ptratio, data = reduced_data)
summary(lm.fit)
lm.fit <- lm(crim ~ rm + rad + ptratio, data = reduced_data)
summary(lm.fit)
final_model <- lm(crim ~ rm + rad, data = reduced_data)
summary(lm.fit)
final_model_df <- reduced_data %>% select(-zn, -chas, -ptratio)

plot(final_model_df)

# (e) 
# All remaining predictors are statistically significant (rm and rad) through use of backwards elimination
# The t-value of rm is -3.78 indicating a negative relationship with crim. this means that an increase in rm should decrease crim.That being said it has a high standard error so it has a large level of uncertainty and is therefore less reliable
# There is a strong positive relationship between rad and crim with a t-value of 16.04 showing that an increase in rad will have a large positive effect on crim. it has a low standard error 

# (f)
# crim =  β0 + β1 ⋅ rm + β2 ⋅ rad  + ϵ

# (g)
# model b has an r^2 value of 0.4922 and adjusted r^2 of 0.4845. this indicates that 48% of the variability in the target (crim) variable can be explained by the predictor variables in this case variability in the target (dependent) variable can be explained by the predictor (independent) variables zn, chas, rm, rad, and ptratio.
# model d has a similar r^2 and adjusted r^2 of 0.4881 and 0.485, respectively. The models have relatively good evidence of correlation

# (h)
#crim = 9.15436 + -1.77340 (rm), crim = 9.15436 + 0.58519 (rad)
confint(final_model, 'rm', level = 0.95)
#the confidence interval for rm is [-2.696, -0.8505]
confint(final_model, 'rad', level = 0.95)
#the confidence interval for rad is [0.5134, 0.6570]

# (i)
?ggplot()

ggplot() + geom_point(aes(final_model_df$rm,final_model_df$crim)) + 
  geom_smooth(aes(final_model_df$rm,final_model_df$crim), method="lm", se=F)

ggplot() + geom_point(aes(final_model_df$rad,final_model_df$crim)) + 
  geom_smooth(aes(final_model_df$rad,final_model_df$crim), method="lm", se=F)

# there is evidence of both outliers and high leverage observations for both predictors.There is very little data that follows the trend.

# (j)
predictors <- as.matrix(final_model_df[, c("rm", "rad")])
models <- list()

for (k in 1:3) {
models[[paste("k=", k)]] <- knn.reg(train = predictors, y = final_model_df$crim, k = k)
}
models

# (k)
?predict()

test_predictions_b <- predict(low_corr, newdata = test_data)
summary(test_predictions_b)

test_predictions_d <- predict(final_model, newdata = test_data)
head(test_predictions_d)

test_predictors <- as.matrix(test_data[, c("rm", "rad")]) 
test_predictions_j <- list()

for (k in 1:3) {
  test_predictions_j[[paste("k=", k)]] <- knn.reg(train = predictors, test = ,  y = final_model_df$crim, k = k)$pred
}
head(test_predictions_j[["k=3"]])
test_predictors


# Task 2 
Boston_2 <- Boston
# (a)
low_percentile <- quantile(Boston_2$crim, 0.25)
high_percentile <- quantile(Boston_2$crim, 0.75)

Boston_2$class <- ifelse(Boston_2$crim > high_percentile, "high", 
                   ifelse(Boston_2$crim < low_percentile, "low", "medium"))
Boston_2$crim <- NULL
head(Boston_2)

# (b)
train_indices <- createDataPartition(
  Boston_2$class,
  times = 1,
  p = 0.66,
  list = FALSE,
)

boston_train <- Boston_2[train_indices , ]
boston_test <- Boston_2[-train_indices , ]

nrow(boston_train)

# (c)

cor_matrix_2 <- cor(boston_train[, !(colnames(boston_train) %in% 'class')])                      
cor_matrix_2
subset_columns <- findCorrelation(
  cor_matrix,
  cutoff = 0.6,
  verbose = FALSE,
  names = TRUE,
)
subset_columns
less_data <- boston_train %>% select(-nox, -indus, -lstat, -tax, -dis)

model_multinom <- multinom(class ~ ., less_data,)

# View the model summary
summary(model_multinom)

knitr::stitch('stats_coursework,knit.r')
