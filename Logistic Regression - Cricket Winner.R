# Question 2

## Part A
```{r ,results='asis' , warning=FALSE, message=FALSE, echo=FALSE}
data_Q2 <- read.csv("Q2dat.csv")
# Remove non informative variables
data_Q2 <- data_Q2[,-(3:4)]
data_Q2 <- data_Q2[,-1]
# Converting categorical data into dummy variables
data_Q2 = makeX(data_Q2)

# Splitting data into 80/20 training test split
indices <- sample(1:nrow(data_Q2), size = 0.8 * nrow(data_Q2))
train_data <- data_Q2[indices, ]
test_data <- data_Q2[-indices, ]

x_train_dummies = as.data.frame(train_data)
x_test_dummies = as.data.frame(test_data)

# Prepare training data for glmnet
x_train <- as.matrix(x_train_dummies[, -which(names(x_train_dummies) == "Defending.Result")])
y_train <- x_train_dummies$Defending.Result

# Define a range of alpha values to explore
alpha_values <- seq(0, 1, by = 0.1)

# Predefine an empty data frame to store results
results <- data.frame(alpha = numeric(), lambda = numeric(), cv_error = numeric())

# Iterate over a sequence of alpha values
for(alpha_val in alpha_values) {
  # Perform cross-validation
  cv_fit <- cv.glmnet(x_train, y_train, family = "binomial", type.measure = "class", alpha = alpha_val)
  
  # Find the lambda that minimizes the cross-validation error for the current alpha
  lambda_min <- cv_fit$lambda.min
  cv_error_min <- min(cv_fit$cvm)
  
  # Append the results for this iteration
  results <- rbind(results, data.frame(alpha = alpha_val, lambda = lambda_min, cv_error = cv_error_min))
}

# Identify the row with the minimum cross-validation error
optimal_row <- which.min(results$cv_error)

# Extract the optimal alpha, lambda, and corresponding cross-validation error
best_alpha <- results$alpha[optimal_row]
best_lambda <- round(results$lambda[optimal_row],3)
best_cv_error <- round(results$cv_error[optimal_row],3)

# Retrain the model using the best alpha and lambda
best_model <- glmnet(x_train, y_train, family = "binomial", alpha = best_alpha, lambda = best_lambda)

# Model coefficients
coeffecients<-coef(best_model, s = best_lambda)
coeffecients_mat <- as.matrix(coeffecients)
coeffecients_df <- as.data.frame(coeffecients_mat)
numeric_cols <- sapply(coeffecients_df, is.numeric)
coeffecients_df[numeric_cols] <- lapply(coeffecients_df[numeric_cols], function(x) format(x, digits = 2, scientific = TRUE))
kable(coeffecients_df, caption = "Coeficients of Logistic Regression Model")

x_test <- as.matrix(x_test_dummies[, -which(names(x_test_dummies) == "Defending.Result")])
predictions <- predict(best_model, newx = x_test, s = best_lambda, type = "response")
predicted_class <- ifelse(predictions > 0.5, 1, 0)

# Calculate Accuracy and F1-Score
#accuracy <- mean(predicted_class == x_train_dummies$Defending.Result)

conf_matrix_caret <- confusionMatrix(as.factor(predicted_class), as.factor(x_test_dummies$Defending.Result))

kable(as.data.frame.matrix(conf_matrix_caret$table), caption = "Confusion Matrix")

stats <- conf_matrix_caret$overall

# Convert to a data frame for pretty printing
stats_df <- data.frame(Statistic = names(stats), Value = as.numeric(stats))

# Use kable() to present the statistics
kable(stats_df, digits = 2, caption = "Confusion Matrix Statistics")

precision <- conf_matrix_caret$byClass['Pos Pred Value']
recall <- conf_matrix_caret$byClass['Sensitivity']

f1_score <- round(2 * (precision * recall) / (precision + recall),2)

# Prepare the predictions and labels for ROC analysis
pred <- prediction(predictions, x_test_dummies$Defending.Result)

# Calculate the ROC curve
roc_perf <- performance(pred, measure = "tpr", x.measure = "fpr")

# Calculate AUC
auc_perf <- performance(pred, measure = "auc")
auc <- round(auc_perf@y.values[[1]],2)
cat("Area under the ROC curve (AUC):", auc, "\n")

# Find the threshold for a recall of at least 0.75
rec_perf <- performance(pred, measure = "sens", x.measure = "cutoff")
rec <- rec_perf@y.values[[1]]
cutoffs <- rec_perf@x.values[[1]]
tau <- round(min(cutoffs[which(rec >= 0.75)]),2)


cat("\nMinimum decision rule threshold $\\tau$ for recall >= 0.75:", tau, "\n")

# Plot the ROC curve
ROC_Plot <- plot(roc_perf, colorize = TRUE)
abline(a = 0, b = 1, lty = 2, col = "gray")
# Add the threshold point to the ROC plot
fpr_perf <- performance(pred, "fpr")
tpr_perf <- performance(pred, "tpr")
fpr <- unlist(fpr_perf@y.values)
tpr <- unlist(tpr_perf@y.values)
specificity <- 1 - fpr
threshold_index <- which.min(abs(cutoffs - tau))
abline(v = fpr[threshold_index], col = "red", lty = 2)
abline(h = tpr[threshold_index], col = "red", lty = 2)
points(fpr[threshold_index], tpr[threshold_index], col = "red", pch = 19)

# Question 3
It should be noted that during data transformation and cleaning the explanatory variables were normalized and “sex” was made into a factor variable. These decisions were taken place to ensure a better more accurate model. Normalizing the data is extremely important in regression models and k-nearest neighbor models as it eliminates the influence of different scales on the model and thus ensures each variable contributes equally to the model.
The target variable (UPDRS) was not normalized as you would need the actual non-normalized score in order to track Parkinson’s disease progression. Thus, normalizing the score would make it somewhat useless.

## Part A

### Linear

The correlation plot depicts the correlation of all the variables with red depicting a strong positive relationship and blue depicting a strong negative relationship.  All variables have a positive correlation with each other with the exception of “HNR” having a negative relationship with all other variables. It should be noted that many variables have a strong correlation with each other; which could be of concern for our regression model and introduce multidisciplinary.

```{r correlation_plot, fig.cap="Correlation Plot",warning=FALSE, message=FALSE, echo=FALSE}
data = read.csv("Q3dat.csv")

correlation_plot <- corrplot(cor(data), 
                             method = 'number', 
                             type = 'upper', 
                             number.cex = 0.5,
                             addCoef.col = "black",
                             col = colorRampPalette(c("blue", "black", "red"))(200),
                             tl.col = "black",
                             tl.cex = 0.5)
```

For the linear model it was decided that a thorough investigation would be performed in order to train the best possible model. We thus made an OLS regression, Ridge regression, Lasso regression and net elastic regression model.

The procedure taken for each linear model, with the exception of the OLS regression model, was somewhat uniform. The OLS model was produced by using all available variables in the data set. We then obtained its RMSE and used it as a benchmark to compare the OLS model to the relative other models. The RMSE scores are used to compare accuracy between different models.

```{r ridge, fig.cap= "Ridge Regression", warning=FALSE, message=FALSE, echo=FALSE}
data$sex <- as.numeric(as.character(data$sex))

numeric_columns <- sapply(data, is.numeric)
columns_to_normalize <- setdiff(names(data)[numeric_columns], "total_UPDRS")

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

```
For the regularized models the coefficients were initially visualized at different levels of the regularization parameter lambda. Thereafter, a cross-validation graph of the respected model was plotted in order to analyse the model’s performance at different levels of lambda. Three distinct lambda values that would best suit the model Were then identified. A lambda value yielding the lowest MSE value, a value 1 standard error away from the lowest MSE and an intermediate between those two. Models were then trained on these lambda value and were then evaluated on their Root Mean Squared Error (RMSE). The model with the lowest RMSE were selected for comparison against other models. The only discrepancy to this method was for the net elastic model, where a loop was created in order to determine the optimal alpha value for the model.
```{r warning=FALSE, message=FALSE, echo=FALSE}
# Convert to a regular matrix (if it's not already) and round the values
combined_ridge_coefs_matrix <- format(round(as.matrix(combined_ridge_coefs), 2), scientific = TRUE)

# Predict at different values of lambda
ridge_min_pred <- predict(ridge_cv, test_x, s = 'lambda.min')
ridge_1se_pred <- predict(ridge_cv, test_x, s = 'lambda.1se')
ridge_exp_pred <- predict(ridge_cv, test_x, s = exp(-2))

# Get MSE
ridge_mse <-vector()
ridge_mse[1] <- round(sqrt(mean((test_y - ridge_min_pred)^2)),2)
ridge_mse[2] <- round(sqrt(mean((test_y - ridge_1se_pred)^2)),2)
ridge_mse[3] <- round(sqrt(mean((test_y - ridge_exp_pred)^2)),2)

```

```{r lasso, fig.cap= "Lasso Regression",warning=FALSE, message=FALSE, echo=FALSE}
par(mfrow=c(1,2))
lasso <- glmnet(x_train, y_train, alpha = 1, standardize = FALSE)
plot(lasso, xvar = 'lambda', label = T)

lasso_cv <- cv.glmnet(x_train, y_train, #this function requires x_train to be a matrix
                      alpha = 1, nfolds = 10, type.measure = 'mse', standardise = FALSE)
plot(lasso_cv)
```

```{r ,warning=FALSE, message=FALSE, echo=FALSE}
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

```

```{r cv_results, fig.cap = "CV of Alpha Value",fig.height = 3 ,warning=FALSE, message=FALSE, echo=FALSE}
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

```
As seen by Figure 12 the alpha value that corresponds to the lowest CV MSE is zero. This indicates that the best Net Elasticity model for the data is actually a Ridge regression model. Thus, there is no need to train a net elasticity model.

```{r warning=FALSE, message=FALSE, echo=FALSE}
df_test_x <- as.data.frame(test_x)
colnames(df_test_x) <- setdiff(names(test_data), "total_UPDRS")
ols_pred <- predict(lm_full, newdata = df_test_x)
#Test MSEs
ols_mse <- round(sqrt(mean((test_y - ols_pred)^2)),2)

min_lasso <- min(lasso_mse)
min_ridge <- min(ridge_mse)
mse_results <- matrix(c(ols_mse, min_ridge, min_lasso), nrow =1)
colnames(mse_results) <- c("OLS RMSE", "Ridge RMSE", "Lasso RMSE")
kable(mse_results, caption = "Comparison of RMSE results")
```

### KNN
```{r knn_fit, fig.cap="CV Results for KNN",results='asis',fig.height = 3,warning=FALSE, message=FALSE, echo=FALSE}
control <- trainControl(method="repeatedcv", number=10, repeats=3)
grid <- expand.grid(k=1:20) # Checking a range of k values from 1 to 20

knn_Fit <- train(total_UPDRS~., data=train_data, method="knn", trControl=control, tuneGrid=grid, preProcess = c("center", "scale"), xlab = "Neighbours")

# Plotting the performance of models with different k values
plot(knn_Fit)
# Training the model with the optimal k found
best_K <- knn_Fit$bestTune$k
```
The K nearest neighbor model is highly dependent on the choice of 'k'. The choice of ‘k’ determines how well the model can generalize from the training data to the unseen data. The optimal value is selected using cross-validation. Based on this we have found the optimal value of k to be `r  best_K`. This can be seen in Figure 13, where the 'k' value of 5 corresponds with the lowest RMSE value of 6.34. 

```{r warning=FALSE, message=FALSE, echo=FALSE}
# Predicting on the test set
predictions_KNN <- predict(knn_Fit, newdata=test_data)

# Evaluating the model
kable(round(postResample(pred = predictions_KNN, obs = test_data$total_UPDRS),2), caption = "Evaluation of KNN")
```

### Random Forest
```{r results='asis',warning=FALSE, message=FALSE, echo=FALSE}
rf_model = randomForest(total_UPDRS~.,data=train_data, mtry = 8,ntree = 250, importance= T, na.action = na.exclude)

# Extract the Mean of Squared Residuals
mean_squared_residuals <- round(rf_model$mse[which.min(rf_model$mse)],2)

# Extract the Percentage of Variance Explained
var_explained <- round(rf_model$rsq[which.max(rf_model$rsq)] * 100, 2)#

# Print the results
cat("Mean of Squared Residuals:", mean_squared_residuals, "\n")
cat("\n% Var Explained:", round(var_explained, 2), "%\n")
```
\newline
For the Random Forest model a thorough investigation took place in order to find the optimal hyper parameters. The selection process involved searching through a wide range of variables tried at each split, result in 6 variables being used. Different number of trees were also used.However, it was found that after 250 trees a further increase in the number of trees being used had no significant effect on the performance of the model. Thus, it was decided that 250 trees was the optimal number of trees for a balance of bias and variance while also keeping run time to sufficient levels.  

Thus, optimal parameters were number of splits 6 and 250 trees. This combination offered the greatest balance of bias and variance in the model. 

```{r warning=FALSE, message=FALSE, echo=FALSE}
predictions_RF <- predict(rf_model, newdata=test_data)

# Evaluating the model

kable(round(postResample(pred = predictions_RF, obs = test_data$total_UPDRS),2),caption = "Evaluation of Random Forest")

```

```{r var_importance_plot,fig.cap="Variable Importance",warning=FALSE, message=FALSE, echo=FALSE}
# Reporting Variable Importance
var_importance <- randomForest::importance(rf_model, type = 1)
var_importance <- var_importance[order(var_importance, decreasing = F), ]
var_importance_plot <- barplot(var_importance, horiz = T, col = 'navy', las = 1,
                               xlab = 'Mean decrease in MSE',cex.names = 0.5)
```
From figure 14 it can be concluded that age is the most important variable in determining UPDRS, followed by DFA and sex.

### Boosted
```{r results='asis',warning=FALSE, message=FALSE, echo=FALSE}

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
```
The development of the Gradient Boosted Model was a lengthy and thorough process. Much of the process involved tinkering with the hyper parameters of the model in order to fine tune the model. It was decided to experiment with different values and combinations of the hyper parameters in order to refine the model’s accuracy. We considered a large spectrum of values for the hyper parameters. The values for n.trees ranged from 1 000 – 100 000; Interaction.depth ranged from 3 – 10; shrinkage ranged from 0.005 – 0.1 and bag fraction from 0.4 – 0.8. After numerous trials and RMSE calculations we decided on an optimal equilibrium for the hyper parameters: 10 000 n.trees, an interaction.depth of 5, a shrinkage of 0.01 and a bag fraction of 0.6. This combination offered the greatest blend of bias and variance, thus leading to the most accurate model.


\newpage
## Part B
```{r RMSE_compare, fig.cap="RMSE Comparison",warning=FALSE,fig.height = 3 ,message=FALSE, echo=FALSE }
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

```
It is seen that all the linear models are approximately performing at similar levels, with an RMSE of approximately 10. KNN was a significant improvement lowering the RMSE to approximately 6.37. Followed by the random forest model improving the RMSE score to 4.1. However, the Gradient Boosted Model should be used on the unseen data. This model has the lowest RMSE, 3.71, relative to all the other models. This means that the GBM should be the best at predicting total UPDRS when compared to the other models.

```{r actual_predicted_plot,fig.cap="Actual vs Predicted UPDRS",warning=FALSE, message=FALSE, echo=FALSE}
std_dev = sd(y_test)

actual_predicted <- data.frame(Actual = y_test, Predicted = yhat_gbm_test)

actual_predicted_plot <-ggplot(actual_predicted, aes(x = Actual, y = Predicted)) +
  geom_point() + 
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") + # Line of perfect prediction
  labs( x = "Actual Values", y = "Predicted Values") +
  theme_minimal() +
  geom_smooth(method = lm, se = FALSE, color = "purple")
plot(actual_predicted_plot)
```
The scatter plot depicts observed data vs predicted data using the GBM. If the GBM model being used was perfect all the observations would be on the 45° line (red dashed line). The purple line is the gradient of our predictions. It is slightly off the red line, indicating that our model predicts scores relatively well. 

The GBM has two major weaknesses. The first weakness is that it is computationally intensive. The model can take long to train and is computationally expensive. This is especially prevalent as the number of trees increases. Secondly, the model is dependent on its hyper parameters. To find the correct combination of these hyper parameters is extremely time consuming and difficult to achieve.


```{r Part C, warning=FALSE, message=FALSE, echo=FALSE}

Q3_testing <- read.csv("Q3testing.csv")
Q3_test_pred <- predict(gbm_lib, newdata = Q3_testing, n.trees = 10000)

# Convert predictions to a data frame
predictions_df <- data.frame(Q3_test_pred)

# Write the predictions to a CSV file
write.csv(predictions_df, "WNRALE002_MTSSAM022.csv", row.names = FALSE, col.names = FALSE, quote = FALSE)
```
