
library(caTools)
library(MASS)

library(drat)

library(xgboost)

setwd("C:/Users/admin/documents")

loan<-read.csv("loan_pre_train.csv",na.strings = c("",NA))
input_test<-read.csv("loan_pre_test.csv",na.strings = c("",NA))

test<-input_test

str(loan)

head(loan)


# data pre processing



#to check number of NAs in each variable
sapply(loan,function(x) sum(is.na(x)))

sapply(test,function(x) sum(is.na(x)))


#-----------------------------------------------------------------------------#
#              pre processing of training data set                            #
#-----------------------------------------------------------------------------#
#replace NAs with mean of contineous variables 

loan$LoanAmount[is.na(loan$LoanAmount)]<-mean(loan$LoanAmount,na.rm = TRUE)
loan$Loan_Amount_Term[is.na(loan$Loan_Amount_Term)]<-mean(loan$Loan_Amount_Term,na.rm = TRUE)

#for categorical variables
loan$Credit_History<-as.factor(loan$Credit_History)

temp<- table(loan$Gender)   #impute the NA values by mode
loan$Gender[is.na(loan$Gender)]<-names(temp[temp==max(temp)])

#imputing Married variable
temp<- table(loan$Married)
loan$Married[is.na(loan$Married)]<-names(temp[temp==max(temp)])

#imputing Dependents

temp<- table(loan$Dependents)
loan$Dependents[is.na(loan$Dependents)]<-names(temp[temp==max(temp)])

#imputing Self_employed

temp<- table(loan$Self_Employed)
loan$Self_Employed[is.na(loan$Self_Employed)]<-names(temp[temp==max(temp)])

temp<- table(loan$Credit_History)
loan$Credit_History[is.na(loan$Credit_History)]<-names(temp[temp==max(temp)])


#-----------------------------------------------------------------------------#
#              pre processing of validation data set                          #
#-----------------------------------------------------------------------------#

#for  continueus variables
test$LoanAmount[is.na(test$LoanAmount)]<-mean(test$LoanAmount,na.rm = TRUE)
test$Loan_Amount_Term[is.na(test$Loan_Amount_Term)]<-mean(test$Loan_Amount_Term,na.rm = TRUE)


#catagorical variables

test$Credit_History<-as.factor(test$Credit_History)

temp<- table(test$Gender)   #impute the NA values by mode
test$Gender[is.na(test$Gender)]<-names(temp[temp==max(temp)])

#imputing Dependents
temp<- table(test$Dependents)
test$Dependents[is.na(test$Dependents)]<-names(temp[temp==max(temp)])

#imputing Self_employed
temp<- table(test$Self_Employed)
test$Self_Employed[is.na(test$Self_Employed)]<-names(temp[temp==max(temp)])

temp<- table(test$Credit_History)
test$Credit_History[is.na(test$Credit_History)]<-names(temp[temp==max(temp)])




#changing level from 3+ to 3 in both training and test data
levels(loan$Dependents)<- c(0,1,2,3)
levels(test$Dependents)<- c(0,1,2,3)
head(loan)
head(test)
# loan$Loan_Amount_Term<-loan$Loan_Amount_Term/12
# 
# test$Loan_Amount_Term<-test$Loan_Amount_Term/12



##Minmax scaling

minmax =function(x){
  xnew<-(x-min(x))/(max(x)-min(x))
}
loan[c(7,8,9,10)]<-apply(loan[c(7,8,9,10)],2,minmax)
test[c(7,8,9,10)]<-apply(test[c(7,8,9,10)],2,minmax)
# loan$intrest<-loan$LoanAmount*loan$Loan_Amount_Term*0.10/(100)

# test<-test[-c(13)]
# test$intrest<-test$LoanAmount*test$Loan_Amount_Term*0.10/(100)

#data partioning
set.seed(101) 


# # install.packages("randomForest")
# library(randomForest)
# 
# library(mlr)
# loan<-createDummyFeatures(loan, cols = "Dependents")
# head(loan)
loan<-loan[,-1 ]
test<-test[,-1]
ind<-sample(1:nrow(loan),round(0.80*nrow(loan)))
dataT<-loan[ind,]
dtest<-loan[-ind,]
# parameters
dataT$Loan_Status<-ifelse(dataT$Loan_Status=="Y",1,0)
dtest$Loan_Status<-ifelse(dtest$Loan_Status=="Y",1,0)

head(dataT)
param<-list("objective"="binary:logistic",
            "eval_metric"="mlogloss",
            "num_class"=12,
            "eta"=0.053,"max.depth"=6,"gamma"=0.025,"subsample"=0.895,
            "colsample_bytree"=0.5225,"min_child_weight"=19)
library(xgboost)


bst.cv=xgb.cv(params=param,data=data.matrix(dataT[,-12]),label=dataT$Loan_Status,nfold=5,nrounds =1000,maximize = FALSE)
##############################################################################


best_param = list()
best_seednumber = 1234
best_logloss = Inf
best_logloss_index = 0

for (iter in 1:100) {
param <- list(objective = "multi:softprob",
eval_metric = "mlogloss",
num_class = 12,
max_depth = sample(6:10, 1),
eta = runif(1, .01, .3),
gamma = runif(1, 0.0, 0.2), 
subsample = runif(1, .6, .9),
colsample_bytree = runif(1, .5, .8), 
min_child_weight = sample(1:40, 1),
max_delta_step = sample(1:10, 1)
)
cv.nround = 1000
cv.nfold = 5
#seednumber = sample.int(10000, 1)[[1]]
set.seed(sample.int(10000, 1)[[1]])
mdcv <- xgb.cv(data=data.matrix(dataT[,-12]),label=dataT$Loan_Status, params = param, nthread=6,
nfold=cv.nfold , nrounds=cv.nround,verbose = T, early_stop_round=8, maximize=FALSE)
mdcv$evaluation_log[,4]
min_logloss =min( mdcv$evaluation_log[,4])
min_logloss_index = which(mdcv$evaluation_log[,4]==min( mdcv$evaluation_log[,4]))

if (min_logloss < best_logloss) {
best_logloss = min_logloss
best_logloss_index = min_logloss_index
best_seednumber = seed.number
best_param = param
}
}







###############################################################################
plot(log((bst.cv$evaluation_log)$test_logloss_mean),type="l")


param<-list("objective"="multi:softprob",
            "eval_metric"=c("logloss","auc"),
            "num_class"=2,
            "eta"=0.053,"max.depth"=6,"gamma"=0.025,"subsample"=0.895,
            "colsample_bytree"=0.5225,"min_child_weight"=19)

bst=xgboost(data=data.matrix(dataT[,-12]),label=dataT$Loan_Status,nrounds=1000,
            nthread=2)

#bstSparse <- xgboost(data = data.matrix(dataT[,-12]), label=dataT$Loan_Status, max.depth = 8, eta = 1, nthread = 2, nround =400 , objective = "binary:logistic")


preds=predict(bst,data.matrix(dtest[,-12]))

prediction <- as.numeric(preds > 0.4)
print(head(prediction))

err <- mean(as.numeric(preds > 0.4) != dtest$Loan_Status)
print(paste("test-error=", err))


library(caret)

confusionMatrix(prediction,dtest$Loan_Status)

print(-mean(log(preds)*dtest$Loan_Status+log(1-preds)*(1-dtest$Loan_Status),na.rm = TRUE))


# # str(dataT)
# 
# 
# model<-randomForest(Loan_Status~.,data=dataT,ntree=1500,na.action = na.fail,replace=TRUE)
# 
# summary(model)
# # plot(model,uniform=TRUE)
# # text(model)
# model$importance
# pred<-predict(model,newdata = dtest,type = "response")
# table(pred,dtest$Loan_Status)
# accuracy<- 1-mean(pred!=dtest$Loan_Status)
# 
# 
# test<-createDummyFeatures(test, cols = "Dependents")
# input_test$Loan_Status<-predict(model,newdata = test,type="response")
# 
# library(caret)
# 
# confusionMatrix(pred,dtest$Loan_Status)
# 
# pred_soln<-input_test[,c(1,13)]
# 
# # str(test)
pred<-predict(bst,data.matrix(test))
pred <- as.numeric(pred > 0.4)
test$Loan_Status<-pred
pred_soln<-test[,c(1,12)]
write.csv(pred_soln,"pred_soln_new.csv",row.names = FALSE)

