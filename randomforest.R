library(needs)
needs(readr,dplyr,ggplot2,corrplot,gridExtra,
      pROC,MASS,caTools,caret, caretEnsemble)

#load data and remove final column of nans
data <- read.csv("C:/Users/anany/Documents/R/kaggle/training_v2.csv")
str(data)
data$hospital_death <- as.factor(data$hospital_death)

summary(data)

library(e1071)
set.seed(123)
linear.tune = tune.svm(hospital_death~., data = data, kernel = "linear", cost = c(1:10))

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

###########################
#Random forest model
rf_model <-train(diagnosis~., data_train, method = "ranger", metric = "ROC", 
                 preProcess = c("center","scale"), trControl = fc)

#random forest prediction and confusion matrix
rf_pred <- predict(rf_model, data_test)
rf_cm <- confusionMatrix(rf_pred, data_test$diagnosis, positive = "M")
rf_cm

#random forest ROC
rf_prediction_prob <- predict(rf_model, data_test, type = "prob")
colAUC(rf_prediction_prob, data_test$diagnosis, plotROC = TRUE)

###########################
#random forest + PCA model
pca_rf_model <- train(diagnosis~., data_train, method = "ranger",mtry <- floor(sqrt(ncol(data_train))/4),
                      metric = "Accuracy", preProcess = c("center","scale","pca"), trControl = fc
                    )

#pca + random forest prediction and confusion matrix
pca_rf_pred <- predict(pca_rf_model, data_test)
pca_rf_cm <- confusionMatrix(pca_rf_pred, data_test$diagnosis, positive = "M")
pca_rf_cm

#pca + random forest ROC
pca_rf_prediction_prob <- predict(pca_rf_model, data_test, type = "prob")
colAUC(pca_rf_prediction_prob, data_test$diagnosis, plotROC = TRUE)

###########################
#classification trees model
clt_model <- train(diagnosis~., data_train, method = "rpart", metric = "ROC",
                   preProcess = c("center","scale"), trControl = fc, tuneLength = 10
                   )

#cl tree prediction and confusion matrix
clt_pred <- predict(clt_model, data_test)
clt_cm <- confusionMatrix(clt_pred, data_test$diagnosis, positive = "M")
clt_cm

#cl tree ROC
clt_prediction_prob <- predict( clt_model, data_test, type = "prob")
colAUC(clt_prediction_prob, data_test$diagnosis, plotROC = TRUE)


###########################
#classification trees + pca model
pca_clt_model <- train(diagnosis~., data_train, method = "rpart", metric = "ROC",
                   preProcess = c("center","scale","pca"), trControl = fc, tuneLength = 10
)

#cl tree + pca prediction and confusion matrix
pca_clt_pred <- predict(pca_clt_model, data_test)
pca_clt_cm <- confusionMatrix(pca_clt_pred, data_test$diagnosis, positive = "M")
pca_clt_cm

#cl tree + pca ROC
pca_clt_prediction_prob <- predict( pca_clt_model, data_test, type = "prob")
colAUC(pca_clt_prediction_prob, data_test$diagnosis, plotROC = TRUE)


###########################
#bagging model