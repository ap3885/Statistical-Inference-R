#k nearest neighbors
library(needs)
needs(readr,dplyr,ggplot2,corrplot,gridExtra,
      pROC,MASS,caTools,caret, caretEnsemble,
      doMC)

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

###################
#knn model
knn_model <- train(diagnosis~., data_train, method = "knn", metric = "ROC",
                   preProcess = c("center","scale"), tuneLength = 10,
                   trControl = fc)

#knn model prediction + confusion matrix
knn_pred <- predict(knn_model, data_test)
knn_cm <- confusionMatrix(knn_pred, data_test$diagnosis, positive = "M")
knn_cm

#polynomial svm ROC
knn_prediction_prob <- predict(knn_model, data_test, type = "prob")
colAUC(knn_prediction_prob, data_test$diagnosis, plotROC = TRUE)

#################
#pca + knn model
pca_knn_model <- train(diagnosis~., data_train, method = "knn", metric = "ROC",
                   preProcess = c("center","scale","pca"), tuneLength = 10,
                   trControl = fc)

#pca + knn model prediction + confusion matrix
pca_knn_pred <- predict(pca_knn_model, data_test)
pca_knn_cm <- confusionMatrix(pca_knn_pred, data_test$diagnosis, positive = "M")
pca_knn_cm

#pca + knn ROC
pca_knn_prediction_prob <- predict(pca_knn_model, data_test, type = "prob")
colAUC(pca_knn_prediction_prob, data_test$diagnosis, plotROC = TRUE)
