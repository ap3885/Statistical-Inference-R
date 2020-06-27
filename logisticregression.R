#logistic regression
library(needs)
needs(readr,dplyr,ggplot2,corrplot,gridExtra,
      pROC,MASS,caTools,caret, caretEnsemble)

#load data and remove final column of nans
data <- read.csv("C:/Users/anany/Documents/R/project/data.csv")
str(data)
data$diagnosis <- as.factor(data$diagnosis)
data[,33] <- NULL
summary(data)

#correlation between data parameters
corr <- cor(data[,3:ncol(data)])
corrplot(corr, order = "hclust", tl.cex = 1, addrect = 8)

#create training and testing partition
set.seed(1)
index <- createDataPartition(data$diagnosis, p = 0.7, list = FALSE)
data_train <- data[index,-1]
data_test <- data[-index,-1]

#training model
fc <- trainControl(method="cv", number = 5, preProcOptions = list(thresh = 0.99), 
                   classProbs = TRUE, summaryFunction = twoClassSummary)

##################
#Logistic regression
lr_model <- train(diagnosis~., data_train, method = "glm", metric = "ROC",
                   preProcess = c("center","scale"), trControl = fc)

#logistic regression model prediction + confusion matrix
lr_pred <- predict(lr_model, data_test)
lr_cm <- confusionMatrix(lr_pred, data_test$diagnosis, positive = "M")
lr_cm

#logistic regression ROC
lr_prediction_prob <- predict(lr_model, data_test, type = "prob")
colAUC(lr_prediction_prob, data_test$diagnosis, plotROC = TRUE)
