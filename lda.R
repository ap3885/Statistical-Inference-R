#LDA
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

#lda
lda_result <- lda(diagnosis~., data = data, center = TRUE, scale = TRUE)
lda_df <- predict(lda_result, data)$x %>% as.data.frame() %>% cbind(diagnosis = data$diagnosis)
lda_result

#plot lda results
ggplot(lda_df, aes(x=LD1, y=0, col = diagnosis)) + geom_point(alpha=0.5)

#plot density of diagnosis column
ggplot(lda_df, aes(x = LD1, fill = diagnosis)) + geom_density(alpha=0.5)

#creating training and testing data from lda
data_train_lda <- lda_df[index,]
data_test_lda <- lda_df[-index,]

#training model
fc <- trainControl(method="cv", number = 5, preProcOptions = list(thresh = 0.99), 
                           classProbs = TRUE, summaryFunction = twoClassSummary)

########################
#lda model
lda_model <- train(diagnosis~., data_train_lda, method = "lda2", metric = "ROC", 
                   PreProc = c("center","scale"),trControl = fc)

#lda prediction and confusion matrix
lda_pred <- predict(lda_model, data_test_lda)
lda_cm <- confusionMatrix(lda_pred, data_test_lda$diagnosis, positive = "M")
lda_cm

#ROC of LDA
lda_prediction_prob <- predict(lda_model, data_test_lda, type = "prob")
colAUC(lda_prediction_prob, data_test_lda$diagnosis, plotROC = TRUE)

########################
#lda + neural nets
lda_nn_model <- train(diagnosis~., data_train_lda, method = "nnet", metric = "ROC",
                      PreProc = c("center","scale"), tuneLength = 10, trace = FALSE,
                      trControl = fc)

#lda + neural nets prediction and confusion matrix
lda_nn_pred <- predict(lda_nn_model, data_test_lda)
lda_nn_cm <- confusionMatrix(lda_nn_pred, data_test_lda$diagnosis, positive = "M")
lda_nn_cm

#ROC of lda + neural nets
lda_nn_prediction_prob <- predict(lda_nn_model, data_test_lda, type = "prob")
colAUC(lda_nn_prediction_prob, data_test_lda$diagnosis, plotROC = TRUE)

########################
#lda + naive bayes
lda_nb_model <- train(diagnosis~., data_train_lda, method = "nb", metric = "ROC",
                      PreProc = c("center","scale"), trace = FALSE, trControl = fc)

#lda + naive bayes prediction and confusion matrix
lda_nb_pred <- predict(lda_nb_model, data_test_lda)
lda_nb_cm <- confusionMatrix(lda_nb_pred, data_test_lda$diagnosis, positive = "M")
lda_nb_cm

#ROC of lda + naive bayes
lda_nb_prediction_prob <- predict(lda_nb_model, data_test_lda, type = "prob")
colAUC(lda_nb_prediction_prob, data_test_lda$diagnosis, plotROC = TRUE)

######################
#lda + Radial SVM
lda_rad_svm_model <- train(diagnosis~., data_train_lda, method = "svmRadial", metric = "ROC",
                       preProcess = c("center","scale"), trace = FALSE, trControl = fc)

#lda + radial svm prediction + confusion matrix
lda_rad_svm_pred <- predict(lda_rad_svm_model, data_test_lda)
lda_rad_svm_cm <- confusionMatrix(lda_rad_svm_pred, data_test_lda$diagnosis, positive = "M")
lda_rad_svm_cm

#lda + radial svm ROC
lda_rad_svm_prediction_prob <- predict(lda_rad_svm_model, data_test_lda, type = "prob")
colAUC(lda_rad_svm_prediction_prob, data_test_lda$diagnosis, plotROC = TRUE)


#######################
#lda + Linear SVM
lda_lin_svm_model <- train(diagnosis~., data_train_lda, method = "svmLinear", metric = "ROC",
                       preProcess = c("center","scale"), trace = FALSE, trControl = fc)

#lda + linear svm prediction + confustion matrix
lda_lin_svm_pred <- predict(lda_lin_svm_model, data_test_lda)
lda_lin_svm_cm <- confusionMatrix(lda_lin_svm_pred, data_test_lda$diagnosis, positive = "M")
lda_lin_svm_cm

#lda + linear svm ROC
lda_lin_svm_prediction_prob <- predict(lda_lin_svm_model, data_test_lda, type = "prob")
colAUC(lda_lin_svm_prediction_prob, data_test_lda$diagnosis, plotROC = TRUE)


#######################
#lda + Polynomial SVM
lda_poly_svm_model <- train(diagnosis~., data_train_lda, method = "svmPoly", metric = "ROC",
                        preProcess = c("center","scale"), trace = FALSE, trControl = fc)

#lda + polynomial svm prediction + confustion matrix
lda_poly_svm_pred <- predict(lda_poly_svm_model, data_test_lda)
lda_poly_svm_cm <- confusionMatrix(lda_poly_svm_pred, data_test_lda$diagnosis, positive = "M")
lda_poly_svm_cm

#lda + polynomial svm ROC
lda_poly_svm_prediction_prob <- predict(lda_poly_svm_model, data_test_lda, type = "prob")
colAUC(lda_poly_svm_prediction_prob, data_test_lda$diagnosis, plotROC = TRUE)

#######################
#lda_knn model
lda_knn_model <- train(diagnosis~., data_train_lda, method = "knn", metric = "ROC",
                   preProcess = c("center","scale"), tuneLength = 10,
                   trControl = fc)

#knn model prediction + confusion matrix
lda_knn_pred <- predict(lda_knn_model, data_test_lda)
lda_knn_cm <- confusionMatrix(lda_knn_pred, data_test_lda$diagnosis, positive = "M")
lda_knn_cm

#polynomial svm ROC
lda_knn_prediction_prob <- predict(lda_knn_model, data_test_lda, type = "prob")
colAUC(lda_knn_prediction_prob, data_test_lda$diagnosis, plotROC = TRUE)

