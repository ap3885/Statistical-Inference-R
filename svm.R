#SVM
library(needs)
needs(readr,dplyr,ggplot2,corrplot,gridExtra,
      pROC,MASS,caTools,caret, caretEnsemble)

#load data and remove final column of nans
data <- read.csv("C:/Users/anany/Documents/R/kaggle/training_v2.csv")
str(data)
data$hospital_death <- as.factor(data$hospital_death)
data[,33] <- NULL
summary(data)

#correlation between data parameters
corr <- cor(data[,3:ncol(data)])
corrplot(corr, order = "hclust", tl.cex = 1, addrect = 8)

#create training and testing partition
set.seed(1)
index <- createDataPartition(data$hospital_death, p = 0.7, list = FALSE)
data_train <- data[index,-1]
data_test <- data[-index,-1]

#training model
fc <- trainControl(method="cv", number = 5, preProcOptions = list(thresh = 0.99), 
                   classProbs = TRUE, summaryFunction = twoClassSummary)


######################
#Radial SVM
rad_svm_model <- train(hospital_death~., data_train, method = "svmRadial", metric = "ROC",
                       trace = FALSE, trControl = fc)

#radial svm prediction + confusion matrix
rad_svm_pred <- predict(rad_svm_model, data_test)
rad_svm_cm <- confusionMatrix(rad_svm_pred, data_test$diagnosis, positive = "M")
rad_svm_cm

#radial svm ROC
rad_svm_prediction_prob <- predict(rad_svm_model, data_test, type = "prob")
colAUC(rad_svm_prediction_prob, data_test$diagnosis, plotROC = TRUE)


#######################
#Linear SVM
lin_svm_model <- train(diagnosis~., data_train, method = "svmLinear", metric = "ROC",
                       preProcess = c("center","scale"), trace = FALSE, trControl = fc)

#linear svm prediction + confustion matrix
lin_svm_pred <- predict(lin_svm_model, data_test)
lin_svm_cm <- confusionMatrix(lin_svm_pred, data_test$diagnosis, positive = "M")
lin_svm_cm

#linear svm ROC
lin_svm_prediction_prob <- predict(lin_svm_model, data_test, type = "prob")
colAUC(lin_svm_prediction_prob, data_test$diagnosis, plotROC = TRUE)


#######################
#Polynomial SVM
poly_svm_model <- train(diagnosis~., data_train, method = "svmPoly", metric = "ROC",
                       preProcess = c("center","scale"), trace = FALSE, trControl = fc)

#polynomial svm prediction + confustion matrix
poly_svm_pred <- predict(poly_svm_model, data_test)
poly_svm_cm <- confusionMatrix(poly_svm_pred, data_test$diagnosis, positive = "M")
poly_svm_cm

#polynomial svm ROC
poly_svm_prediction_prob <- predict(poly_svm_model, data_test, type = "prob")
colAUC(poly_svm_prediction_prob, data_test$diagnosis, plotROC = TRUE)

###missing data

library(mice)

md.pattern(data)
