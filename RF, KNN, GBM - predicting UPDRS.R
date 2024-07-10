# Question 3
## Part A

### Linear Regression Model

data = read.csv("Q3dat.csv")

correlation_plot <- corrplot(cor(data), 
                             method = 'number', 
                             type = 'upper', 
                             number.cex = 0.5,
                             addCoef.col = "black",
                             col = colorRampPalette(c("blue", "black", "red"))(200),
                             tl.col = "black",
                             tl.cex = 0.5)


data$sex <- as.numeric(as.character(data$sex))

numeric_columns <- sapply(data, is.numeric)
columns_to_normalize <- setdiff(names(data)[numeric_columns], "total_UPDRS")
# clean and process data
preProcValues <- preProcess(data[, columns_to_normalize], method = c("center", "scale"))
dataNorm <- predict(preProcValues, data[, columns_to_normalize])
UPDRS <- data[,"total_UPDRS"]
data[, columns_to_normalize] <- dataNorm
data[,"total_UPDRS"] <- UPDRS

# test and training data
indices <- sample(1:nrow(data), size = 0.8 * nrow(data))
train_data <- data[indices, ]
test_data <- data[-indices, ]

# linear model with variable selection
x_train <- as.matrix(train_data[, setdiff(names(train_data), "total_UPDRS")])
y_train <- train_data$total_UPDRS

test_x <- as.matrix(test_data[, setdiff(names(test_data), "total_UPDRS")])
test_y <- test_data$total_UPDRS

#Linear Model
# Fit lm using all features except 'total_UPDRS'
lm_full <- lm(total_UPDRS ~ ., data = train_data)


par(mfrow=c(1,2))
# peform ridge regularization
ridge <- glmnet(x_train, y_train, alpha = 0, standardize = FALSE,
                lambda = exp(seq(-4, 5, length.out = 100)))
plot(ridge, xvar = 'lambda', label = T)

#Apply 10-fold CV
ridge_cv <- cv.glmnet(x_train, y_train, 
                      alpha = 0, nfolds = 10, type.measure = 'mse', standardise = FALSE,
                      lambda = exp(seq(-4, 5, length.out = 100))) #Default lambda range doesn't cover minimum
plot(ridge_cv)

abline(h = ridge_cv$cvup[which.min(ridge_cv$cvm)], lty = 2)

combined_ridge_coefs <- cbind(
  r_lambda_min = coef(ridge_cv, s = 'lambda.min'), ## Lamda min
  r_lambda_1se = coef(ridge_cv, s = 'lambda.1se'), ## lamda 1 se
  r_lambda_exp_2 = coef(ridge_cv, s = exp(-2))  ## lamda in between
)

# Convert to a regular matrix (if it's not already) and round the values
combined_ridge_coefs_matrix <- format(round(as.matrix(combined_ridge_coefs), 2), scientific = TRUE)

# Predict at different values of lambda
ridge_min_pred <- predict(ridge_cv, test_x, s = 'lambda.min')
ridge_1se_pred <- predict(ridge_cv, test_x, s = 'lambda.1se')
ridge_exp_pred <- predict(ridge_cv, test_x, s = exp(-2))

# Get MSE to find best alpha
ridge_mse <-vector()
ridge_mse[1] <- round(sqrt(mean((test_y - ridge_min_pred)^2)),2)
ridge_mse[2] <- round(sqrt(mean((test_y - ridge_1se_pred)^2)),2)
ridge_mse[3] <- round(sqrt(mean((test_y - ridge_exp_pred)^2)),2)

par(mfrow=c(1,2))
lasso <- glmnet(x_train, y_train, alpha = 1, standardize = FALSE)
plot(lasso, xvar = 'lambda', label = T)

lasso_cv <- cv.glmnet(x_train, y_train, #this function requires x_train to be a matrix
                      alpha = 1, nfolds = 10, type.measure = 'mse', standardise = FALSE)
plot(lasso_cv)

combined_lasso_coefs <- cbind(
  l_lambda_min = coef(lasso_cv, s = 'lambda.min'), ## Lamda min
  l_lambda_1se = coef(lasso_cv, s = 'lambda.1se'), ## lamda 1 se
  l_lambda_exp_5 = coef(ridge_cv, s = exp(-5))  ## lamda in between
)

# Convert to a regular matrix (if it's not already) and round the values
combined_lasso_coefs_matrix <- format(round(as.matrix(combined_lasso_coefs), 2), scientific = TRUE)

lasso_min_pred <- predict(lasso_cv, test_x, s = 'lambda.min')
lasso_1se_pred <- predict(lasso_cv, test_x, s = 'lambda.1se')
lasso_exp_pred <- predict(lasso_cv, test_x, s = exp(-5))

lasso_mse <- vector()
lasso_mse[1] <- round(sqrt(mean((test_y - lasso_min_pred)^2)),2)
lasso_mse[2] <- round(sqrt(mean((test_y - lasso_1se_pred)^2)),2)
lasso_mse[3] <- round(sqrt(mean((test_y - lasso_exp_pred)^2)),2)

# Net Regularization
alpha_values <- seq(0, 1, by=0.1)
cv_results <- list()

# Test for different values of alpha with CV
for(i in alpha_values) {
  cv_fit <- cv.glmnet(x_train, y_train, 
                      alpha = i, 
                      nfolds = 10, 
                      type.measure = 'mse', 
                      standardize = FALSE, 
                      lambda = exp(seq(-4, 5, length.out = 100)))
  
  cv_results[[paste("alpha", i)]] <- cv_fit
}

# Depict the MSE visually so we can choose the correct alpha
plot_alpha_performance <- function(cv_results) {
  mse_values <- sapply(cv_results, function(x) min(x$cvm))
  alpha_values_numeric <- seq(0, 1, by=0.1)
  plot(alpha_values_numeric, mse_values, type = 'b', pch=19, xlab = "Alpha", ylab = "CV MSE", xlim=c(0, 1))
}

plot_alpha_performance(cv_results)

df_test_x <- as.data.frame(test_x)
colnames(df_test_x) <- setdiff(names(test_data), "total_UPDRS")
ols_pred <- predict(lm_full, newdata = df_test_x)
#Test MSE's
ols_mse <- round(sqrt(mean((test_y - ols_pred)^2)),2)

min_lasso <- min(lasso_mse)
min_ridge <- min(ridge_mse)
mse_results <- matrix(c(ols_mse, min_ridge, min_lasso), nrow =1)
colnames(mse_results) <- c("OLS RMSE", "Ridge RMSE", "Lasso RMSE")
kable(mse_results, caption = "Comparison of RMSE results")
# use regularization with lowest MSE

### K-Nearest Neighbour Model
```{r knn_fit, fig.cap="CV Results for KNN",results='asis',fig.height = 3,warning=FALSE, message=FALSE, echo=FALSE}
control <- trainControl(method="repeatedcv", number=10, repeats=3)
grid <- expand.grid(k=1:20) # Checking a range of k values from 1 to 20

knn_Fit <- train(total_UPDRS~., data=train_data, method="knn", trControl=control, tuneGrid=grid, preProcess = c("center", "scale"), xlab = "Neighbours")

# Plotting the performance of models with different k values
plot(knn_Fit)
# Training the model with the optimal k found
best_K <- knn_Fit$bestTune$k

# Predicting on the test set
predictions_KNN <- predict(knn_Fit, newdata=test_data)

# Evaluating the model
kable(round(postResample(pred = predictions_KNN, obs = test_data$total_UPDRS),2), caption = "Evaluation of KNN")

### Random Forest Model
rf_model = randomForest(total_UPDRS~.,data=train_data, mtry = 8,ntree = 250, importance= T, na.action = na.exclude)

# Extract the Mean of Squared Residuals
mean_squared_residuals <- round(rf_model$mse[which.min(rf_model$mse)],2)

# Extract the Percentage of Variance Explained
var_explained <- round(rf_model$rsq[which.max(rf_model$rsq)] * 100, 2)#

# Print the results
cat("Mean of Squared Residuals:", mean_squared_residuals, "\n")
cat("\n% Var Explained:", round(var_explained, 2), "%\n")

predictions_RF <- predict(rf_model, newdata=test_data)

# Evaluating the model

kable(round(postResample(pred = predictions_RF, obs = test_data$total_UPDRS),2),caption = "Evaluation of Random Forest")

# Reporting Variable Importance
var_importance <- randomForest::importance(rf_model, type = 1)
var_importance <- var_importance[order(var_importance, decreasing = F), ]
var_importance_plot <- barplot(var_importance, horiz = T, col = 'navy', las = 1,
                               xlab = 'Mean decrease in MSE',cex.names = 0.5)

### Gradient Boosted Model

gbm_lib <- gbm(total_UPDRS ~ ., data = train_data, 
               distribution = 'gaussian', #Applies squared error loss 
               n.trees = 10000,
               interaction.depth = 5,
               shrinkage = 0.01,
               bag.fraction = 0.6)          #Subsamples observations by default (0.5)

yhat_gbm_test <- predict(gbm_lib, newdata = test_data, n.trees = 10000)


# Extract actual target values from test_data
y_test <- test_data[, 3]

# Calculate the mean squared error for the test data
mse_gbm_test <- round(sqrt(mean((y_test - yhat_gbm_test)^2)),2)

cat("The RMSE for the gradient boosted model is", mse_gbm_test)

## compare the different models to find the best one

# Calculate RMSE for KNN (assuming predictionsKNN are predictions from the KNN model)
rmseKNN <- round(sqrt(mean((predictions_KNN - test_data$total_UPDRS)^2)),2)

# Calculate RMSE for Random Forest (assuming predictionsRF are predictions from the RF model)
rmseRF <- round(sqrt(mean((predictions_RF - test_data$total_UPDRS)^2)),2)

RMSE_compare <- as.matrix(c(mse_results, rmseKNN, rmseRF, mse_gbm_test),nrow(1))
RMSE_compare <- t(RMSE_compare)
colnames(RMSE_compare) <- c("OLS", "Ridge","Lasso", "KNN","RF","GBM")

# Convert the matrix to a data frame for plotting
RMSE_df <- as.data.frame(RMSE_compare)
RMSE_mat <-as.matrix(RMSE_compare)

# Reshape the data frame from wide to long format
library(tidyr)
df_long <- pivot_longer(RMSE_df, cols = everything(), names_to = "Method", values_to = "Value")
df_long$Method <- factor(df_long$Method, levels = df_long$Method[order(-df_long$Value)])

# View the reshaped data frame
kable(df_long, caption = "RMSE Comparison of the Models")

RMSE_compare <- barplot(RMSE_mat, horiz = T, col = 'navy', las = 1,
                        xlab = 'RMSE',ylab = "Method",cex.names = 0.6)

std_dev = sd(y_test)

actual_predicted <- data.frame(Actual = y_test, Predicted = yhat_gbm_test)

actual_predicted_plot <-ggplot(actual_predicted, aes(x = Actual, y = Predicted)) +
  geom_point() + 
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") + # Line of perfect prediction
  labs( x = "Actual Values", y = "Predicted Values") +
  theme_minimal() +
  geom_smooth(method = lm, se = FALSE, color = "purple")
plot(actual_predicted_plot)
