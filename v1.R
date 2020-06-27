library(needs)
needs(readr,dplyr,ggplot2,corrplot,gridExtra,
      pRoc,MASS,caTools,caret, caretEnsemble,
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

#lda model
lda_model <- train(diagnosis~., data_train_lda, method = "lda2", metric = "ROC", 
                   PreProc = c("center","scale"),trControl = fc)

#lda prediction and confusion matrix
lda_pred <- predict(lda_model, data_test_lda)
lda_cm <- con
