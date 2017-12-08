getwd()
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
library(caTools)
library(MASS)
# install.packages("randomForest")
library(randomForest)


loan<-loan[,-1 ]
test<-test[,-1]
ind<-sample(1:nrow(loan),round(0.85*nrow(data),2))
dataT<-loan[ind,]
dtest<-loan[-ind,]

# str(dataT)

model<-randomForest(Loan_Status~.,data=dataT,ntree=1000,na.action = na.fail,replace=TRUE)
summary(model)
# plot(model,uniform=TRUE)
# text(model)
model$importance
pred<-predict(model,newdata = dtest,type = "response")
table(pred,dtest$Loan_Status)
accuracy<- 1-mean(pred!=dtest$Loan_Status)



input_test$Loan_Status<-predict(model,newdata = test,type="response")

pred_soln<-input_test[,c(1,13)]

# str(test)
# pred<-predict(model,test)
# test$Loan_Status<-pred$class
# pred_soln<-test[,c(1,13)]
write.csv(pred_soln,"pred_soln_new.csv",row.names = FALSE)

