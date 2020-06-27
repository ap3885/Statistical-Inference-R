#PCA
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

#PCA
pca_result <- prcomp(data[,3:ncol(data)], center = TRUE, scale = TRUE)
plot(pca_result, type = "l")
summary(pca_result)
pca_df <- as.data.frame(pca_result$x)
ggplot(pca_df, aes(x = PC1, y = PC2, col = data$diagnosis)) + geom_point(alpha=0.5)

#plot first 2 PCs
pc1_g <- ggplot(pca_df, aes(x = PC1, fill = data$diagnosis)) + geom_density(alpha=0.25)
pc2_g <- ggplot(pca_df, aes(x = PC2, fill = data$diagnosis)) + geom_density(alpha=0.25)
grid.arrange(pc1_g, pc2_g, ncol = 2)

#training model
fc <- trainControl(method="cv", number = 5, preProcOptions = list(thresh = 0.99), 
                   classProbs = TRUE, summaryFunction = twoClassSummary)

###############
#pca + neural nets
pca_nn_model <- train(diagnosis~., data_train, method = "nnet", metric = "ROC",
                      preProcess = c("center","scale"), tuneLength = 10,
                      trace = FALSE, trControl = fc)

#pca + neural nets prediction and confusion matrix
pca_nn_pred <- predict(pca_nn_model, data_test)
pca_nn_cm <- confusionMatrix(pca_nn_pred, data_test$diagnosis, positive = "M")
pca_nn_cm

#ROC of PCA + neural nets
pca_nn_prediction_prob <- predict(pca_nn_model, data_test, type = "prob")
colAUC(pca_nn_prediction_prob, data_test$diagnosis, plotROC = TRUE)

###############
########################
#PCA + naive bayes
pca_nb_model <- train(diagnosis~., data_train, method = "nb", metric = "ROC",
                      PreProc = c("center","scale"), trace = FALSE, trControl = fc)

#PCA + naive bayes prediction and confusion matrix
pca_nb_pred <- predict(pca_nb_model, data_test)
pca_nb_cm <- confusionMatrix(pca_nb_pred, data_test$diagnosis, positive = "M")
pca_nb_cm

#PCA of lda + naive bayes
pca_nb_prediction_prob <- predict(pca_nb_model, data_test, type = "prob")
colAUC(pca_nb_prediction_prob, data_test$diagnosis, plotROC = TRUE)

