
################################
# Install packages and libraries
################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(klaR)) install.packages("klaR", repos = "http://cran.us.r-project.org")
if(!require(nnet)) install.packages("nnet", repos = "http://cran.us.r-project.org")
library(rmarkdown)
library(ggplot2)
library(lubridate)
library(tidyverse) 
library(corrplot)
library(caret)
library(nnet)

################################
# Load dataset
################################

data <- read.csv("https://raw.githubusercontent.com/happycheers/BreastCancer/master/data.csv")

################################
# Data exploration
################################

#Structure of the dataset
str(data)

# First 6 rows and header 
head(data)

# Summary of statitics
summary(data)

# Summarize number of diagnosis ("B" and "M") in the dataset
data %>% group_by(diagnosis) %>% summarize(n())

################################
# Data cleaning
################################

# Remove columns 1 and 33 as irrelevant
data <- data[,-33]
data <- data[,-1]

# Check if there are missing values
map_int(data, function(.x) sum(is.na(.x)))


# Plot the correlation among variables
corrplot(cor(data[,2:31]) , main=" Corrplot" , method = "circle" , type = "upper")

# Identify variables with correlation coefficient higher than 0.9 or lower than -0.9
to_drop_col <- findCorrelation(cor(data[,2:31]), cutoff=0.9)

# Adjust the result by one column shift
to_drop_col <- to_drop_col + 1

# Remove highly correlated variables
new_data <- data[,-to_drop_col]

# Cross-check if highly correlated variables have been removed
findCorrelation(cor(new_data[,2:21]), cutoff=0.9)

##################################
# Create training and testing sets
##################################

# Divide the data set into training (80%) and testing (20%) sets
set.seed(1234, sample.kind="Rounding")
index <- createDataPartition(new_data$diagnosis, times=1, p=0.8, list = FALSE)
train <- new_data[index, ]
test <- new_data[-index, ]

##################################
# Data Analysis - Modelling Approach
##################################

# Cross validatin with 10 folds
tc <- trainControl(method="cv", number = 10, classProbs=TRUE, summaryFunction = twoClassSummary)

################################
# Naive bayes model
################################

# Train a naive bayes model
naiveb_model <- train(diagnosis~., 
                      train, 
                      method="nb", 
                      metric = "ROC",  
                      preProcess=c('center','scale'), 
                      trControl=tc)

# Predict testing set
naiveb_pred <- predict(naiveb_model, test)

# summarize results (set positive as "M" so that the sensitivity is correct)
naiveb_result <- confusionMatrix(naiveb_pred, test$diagnosis, positive = "M")
naiveb_result

################################
# Logistic regression model
################################

# Train a logistic regression model
glm_model <- train(diagnosis~., 
                   train, 
                   method="glm", 
                   metric = "ROC",  
                   preProcess=c('center','scale'), 
                   trControl=tc)

# Predict testing set
glm_pred <- predict(glm_model, test)

# summarize results (set positive as "M" so that the sensitivity is correct)
glm_result <- confusionMatrix(glm_pred, test$diagnosis, positive = "M")
glm_result 

################################
# K-nearest neighbor model
################################

# Train a KNN model
knn_model <- train(diagnosis~., 
                   train, 
                   method="knn", 
                   metric = "ROC",  
                   preProcess=c('center','scale'), 
                   tuneLength=10,
                   trControl=tc)

# Predict testing set
knn_pred <- predict(knn_model, test)

# summarize results (set positive as "M" so that the sensitivity is correct)
knn_result <- confusionMatrix(knn_pred, test$diagnosis, positive = "M")
knn_result 

################################
# Random forest model
################################

# Train a random forest model
rf_model <- train(diagnosis~., 
                  train, 
                  method="rf", 
                  metric = "ROC",  
                  preProcess=c('center','scale'), 
                  trControl=tc)

# Predict testing set
rf_pred <- predict(rf_model, test)

# summarize results (set positive as "M" so that the sensitivity is correct)
rf_result <- confusionMatrix(rf_pred, test$diagnosis, positive = "M")
rf_result 

################################
# Results
################################

# Summarize the confusion matrixes of each model
result_list <- list (naive_bayes = naiveb_result,
                     logistic_regression = glm_result,
                     KNN = knn_result,
                     random_forest = rf_result)
results <- sapply (result_list, function(x) x$byClass)

# Print the results in a table format
results %>% knitr::kable()

# Identify the best results for each metric in confusion matrix
best_results <- apply(results, 1, which.is.max)

# Match the best results with corresponding model
report <- tibble (metric = names(best_results),
                  best_model = colnames(results)[best_results],
                  value=mapply(function(x,y) {results[x,y]},
                               names(best_results),
                               best_results))
rownames(report)<-NULL

# Print the best model identified for each metric
report

