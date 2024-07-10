library(kableExtra)
library(sf)
library(tree)
library(caret)
library(dplyr)
library(ROCR)
library(MLmetrics)
library(class)
library(ggplot2)
library(randomForest)
library(corrplot)
library(kableExtra)
library(broom)
library(glmnet)
library(rpart)
library(gbm)
library(knitr)
set.seed(123)


# Question 1
## Part A

data_Q1 <- read.csv("Q1dat.csv")
data_Q1$quality <- as.factor(data_Q1$quality)

# Training and testing data
indices <- sample(1:nrow(data_Q1), size = 0.8 * nrow(data_Q1))
train_data <- data_Q1[indices, ]
test_data <- data_Q1[-indices, ]


# Adjust tree control settings for maximum overfitting
control_settings <- tree.control(nobs = nrow(train_data), mincut = 1, minsize = 2, mindev = 0)

# Fit the tree with adjusted control settings
mealie_tree_gini <- tree(quality ~ ., data = train_data, control = control_settings)
plot(mealie_tree_gini)
text(mealie_tree_gini, pretty = 0)

predictions_mealie <- predict(mealie_tree_gini, test_data, type = "class")
actual_class <- test_data$quality
missclass <- round(sum(predictions_mealie != actual_class) / length(actual_class),2)

cat("The testing missclassification rate is", missclass, ". This indicates that the over fitted tree is able to classify mealie quality well. \n")

## Part B
mealie_cv <- cv.tree(mealie_tree_gini, FUN = prune.misclass) #Use classification error rate for pruning

# Make the CV plot
plot(mealie_cv$size, mealie_cv$dev, type = 'o',
     pch = 16, col = 'navy', lwd = 2,
     xlab = 'Number of terminal nodes', ylab='CV error')


alpha <- round(mealie_cv$k,1)
axis(3, at = mealie_cv$size, lab = alpha, cex.axis = 0.8)
mtext(expression(alpha), 3, line = 2.5, cex = 1.2)
axis(side = 1, at = 1:max(mealie_cv$size))

# Getting size of tree
# 1 se rule
min_cv_error <- min(mealie_cv$dev)
min_error_index <- which.min(mealie_cv$dev)
min_error_size <- mealie_cv$size[min_error_index]

# Calculate the standard error of the minimum cross-validation error
std_error <- sd(mealie_cv$dev) / sqrt(length(mealie_cv$dev))

# Find the largest tree size within one standard error of the minimum error
upper_bound <- min_cv_error + std_error
simpler_models <- which(mealie_cv$dev <= upper_bound)
if (length(simpler_models) > 0) {
  # Select the simplest model within the bound, which might be the one with the minimum size
  optimal_size_1se <- min(mealie_cv$size[simpler_models])
} else {
  optimal_size_1se <- min_error_size
}

# min 1se tree
mealie_pruned_1se <- prune.misclass(mealie_tree_gini, best = optimal_size_1se)
plot(mealie_pruned_1se)
text(mealie_pruned_1se, pretty = 0)

actual_class <- test_data$quality

predictions_mealie_1se <- predict(mealie_pruned_1se, test_data, type = "class")
missclass_1se <- round(sum(predictions_mealie_1se != actual_class) / length(actual_class),2)

cat("The model missclassification rate is", missclass_1se, ".\n")

## Part C

df <- train_data
df_plot <- data.frame(longitude = train_data$longitude, latitude = train_data$latitude, quality = train_data$quality)

# Using base R to plot, color-coded by 'quality'
plot(df_plot$longitude, df_plot$latitude, col=df_plot$quality, xlab="Longitude", ylab="Latitude", pch=20)
legend("topright", legend=unique(df_plot$quality), col=unique(df_plot$quality), pch=20)
```

# rotate area for better tree split
rotated_data <- function(train_data, theta) {
  rotation_matrix <- matrix(c(cos(theta), -sin(theta), sin(theta), cos(theta)), nrow = 2)
  return(as.data.frame(t(rotation_matrix %*% t(train_data))))
}

thetas <- seq(0, pi/2, by = 0.01*pi)
# Placeholder for storing results
cv_results <- data.frame(theta = thetas, MisclassificationRate = rep(NA, length(thetas)))

# manually make cross validation
K <- 10
train_data$fold <- sample(1:K, nrow(train_data), replace = TRUE)

for(i in seq_along(thetas)) {
  theta <- thetas[i]
  misclass_rates <- numeric(K) # To store misclassification rates for each fold
  
  # Split the data into training and testing based on the fold
  for(k in 1:K) {
    test_indices <- which(train_data$fold == k)
    train_indices <- setdiff(1:nrow(train_data), test_indices)
    
    # Apply rotation
    rotated_df <- train_data %>%
      select(longitude, latitude) %>%
      rotated_data(theta) %>%
      bind_cols(train_data %>% select(-longitude, -latitude), .)
    
    names(rotated_df)[1:6] <- c("Count", "Pests", "Height", "Quality", "Longitude", "Latitude")
    
    # Splitting rotated data into training and testing sets
    rotated_train_data <- rotated_df[train_indices, ]
    rotated_test_data <- rotated_df[test_indices, ]
    
    # Training the tree model
    fit <- tree(Quality ~ ., data = rotated_train_data)
    
    # Predictions and misclassification rate calculation
    predict_rotated <- predict(fit, newdata = rotated_test_data, type = "class")
    actual_class <- rotated_test_data$Quality
    misclass_rates[k] <- sum(predict_rotated != actual_class) / length(actual_class)
  }
  
  # Storing the average misclassification rate for the current theta
  cv_results$MisclassificationRate[i] <- mean(misclass_rates)
}

# Remove the fold column not to pollute your dataset
train_data$fold <- NULL
# Plotting CV misclassification rate as a function of θ
plot(cv_results$theta, cv_results$MisclassificationRate, type="b", xlab="Theta (Radians)", ylab="CV Misclassification Rate")


# Identify the best rotation (θ)
best_theta <- round(cv_results$theta[which.min(cv_results$MisclassificationRate)],2)
cat("The optimal angle of rotation is", best_theta, "radians.\n")

rotated_df <- data_Q1 %>%
  select(longitude, latitude) %>%
  rotated_data(best_theta) %>% 
  bind_cols(data_Q1 %>% select(-longitude, -latitude), .)

# Correct column names
names(rotated_df)[1:6] <- c("Count", "Pests", "Height", "Quality", "Longitude", "Latitude")

rotated_train_data <- rotated_df[indices, ]
rotated_test_data <- rotated_df[-indices, ]

best_fit <- tree(Quality ~ ., data = rotated_train_data)
plot(best_fit)
text(best_fit)

predict_best_fit <- predict(best_fit, newdata = rotated_test_data, type = "class")
actual_class <- rotated_test_data$Quality
missclass_rotated <- sum(predict_best_fit != actual_class) / length(actual_class)

cat("The rotated missclassification rate is", missclass_rotated, ".\n")

df_plot <- data.frame(longitude = rotated_train_data$Longitude, latitude = rotated_train_data$Latitude, quality = rotated_train_data$Quality)

# Using base R to plot, color-coded by 'quality'
plot(df_plot$longitude, df_plot$latitude, col=df_plot$quality, main="Quality vs. Location", xlab="Longitude", ylab="Latitude", pch=20)
legend("topright", legend=unique(df_plot$quality), col=unique(df_plot$quality), pch=20)


