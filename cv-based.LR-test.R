# This script applies a cv-based LR-test using the Boston housing dataset.
# The approach is illustrated using a linear model, gbm and xgboost.
# The approach by Dezeure et al (2015), https://arxiv.org/abs/1408.4026, implemented in the hdi package, is applied as well.
library(MASS)
library(lmtest)
library(dplyr)
library(caret)
library(ggplot2)
library(tree)
library(randomForest)
library(gbm)
library(xgboost)
library(hdi)

# data ####
data(Boston)
data=Boston

model.lm=lm(medv~.,data=data)
summary(model.lm)

depvar="medv"
testvar="age" 
regvars.0=names(data)
regvars.0=regvars.0[!regvars.0 %in% c(depvar,testvar)] # regvars.0 includes all variables except depvar and testvar 
regvars.1=c(regvars.0,"testvar.rand") # 'testvar.rand' is the name of the scrambled version of 'testvar' (see below)
regvars.2=c(regvars.0,testvar) # the complete list of regressors

# tune tree models with caret; use 'larger' model, i.e. all regressors (regvars.2) for that purpose

train.ctrl <- trainControl(
  number=5,
  method="repeatedcv",
  repeats=5, 
  verboseIter=F
) 

# note: the tunegrids for gbm and xgb may require to be changed when another testvar is chosen! 
gbm.tunegrid <- expand.grid(interaction.depth=c(1,2), # default=1
                            n.trees=c(500,1000,1500,2000,2500,3000), # default=100 
                            shrinkage=c(0.05,0.1,0.15), # default=0.1
                            n.minobsinnode=10) # default=10

xgb.tunegrid <- expand.grid(
  nrounds = 200,
  max_depth = c(4,6,8), # default 6
  #max_depth = 4, # default 6
  eta = c(0.1,0.3,0.5), # ctrl learning rate; default 0.3; low=robust to overfitting
  #eta = 0.1, # ctrl learning rate; default 0.3; low=robust to overfitting
  gamma = c(0,1), # default 0; large=more conservative
  colsample_bytree = 1, # default 1
  subsample = 1, # ratio of the training instance; default 1; low=prevents overfitting
  min_child_weight = 1  # default 1; large=more conservative
)

set.seed(76543)
model.gbm <- train(
  x = data[,regvars.2],
  y = data[,depvar],
  method = 'gbm', 
  metric="RMSE",
  trControl = train.ctrl,
  verbose=F,
  tuneGrid = gbm.tunegrid
)
plot(model.gbm)
#print(model.gbm)
print(varImp(model.gbm, scale = TRUE))

set.seed(87654)
model.xgb <- train(
  x = data[,regvars.2],
  y = data[,depvar],
  method = 'xgbTree', 
  metric="RMSE",
  trControl = train.ctrl,
  tuneGrid = xgb.tunegrid
)
plot(model.xgb)
#print(model.xgb)
print(varImp(model.xgb, scale = TRUE))

# functions ####

# gf (goodness of fit measures) 
gf=function(method,model,y,x,n.trees=0){

  if(method=="gbm"){
    yhat=predict(model,newdata=x,n.trees)
  }else{
    yhat=predict(model,newdata=x)
  }
  ybar=mean(y,na.omit=T)
  SST=sum((y-ybar)^2)
  SSR=sum((yhat-ybar)^2)
  SSE=sum((y-yhat)^2)
  
  Rsq=SSR/SST
  Rsq.uncentered=sum(yhat^2)/sum(y^2)
  Rsq.multiple=cor(y,yhat)^2
  
  return(c("Rsq"=Rsq,"Rsq.multiple"=Rsq.multiple,"Rsq.uncentered"=Rsq.uncentered,"SSE"=SSE))
}

estimate=function(method,depvar,regvars,data.train,data.valid){
  
  frml=as.formula(paste0(depvar,"~",paste0(regvars,collapse="+")))
  
  if(method=="lm"){
    model=lm(frml,data=data.train)
  }
  
  if(method=="tree"){
    model=tree(frml,data=data.train)
  }
  
  if(method=="rf"){
    model=randomForest(frml,data=data.train)
  }
  
  if(method=="lm" | method=="tree" | method=="rf"){
    #print(summary(model))
    gf.ins=gf(method,model,data.train[,depvar],data.train)
    if(is.null(data.valid)==F){
      gf.oos=gf(method,model,data.valid[,depvar],data.valid)  
    }else{
      gf.oos=NULL
    }
  }
  
  if(method=="gbm" | method=="xgb"){
    
    if(method=="gbm"){
      model=gbm(frml,data=data.train,distribution="gaussian",
                n.trees=model.gbm$bestTune$n.trees,
                shrinkage=model.gbm$bestTune$shrinkage,
                interaction.depth=model.gbm$bestTune$interaction.depth,
                n.minobsinnode=model.gbm$bestTune$n.minobsinnode)
      gf.ins=gf(method,model,data.train[,depvar],data.train,n.trees=model.gbm$bestTune$n.trees)
      if(is.null(data.valid)==F){
        gf.oos=gf(method,model,data.valid[,depvar],data.valid,n.trees=model.gbm$bestTune$n.trees)  
      }else{
        gf.oos=NULL
      }
    }
    
    if(method=="xgb"){
      model <- xgboost(data = as.matrix(data.train[,regvars]), label = data.train[,depvar] ,verbose=F,
                       nrounds=model.xgb$bestTune$nrounds,
                       max_depth=model.xgb$bestTune$max_depth,
                       eta=model.xgb$bestTune$eta,
                       gamma=model.xgb$bestTune$gamma,
                       colsample_bytree=model.xgb$bestTune$colsample_bytree,
                       subsample=model.xgb$bestTune$subsample,
                       min_child_weight=model.xgb$bestTune$min_child_weight)
      gf.ins=gf(method,model,data.train[,depvar],as.matrix(data.train[,regvars]))
      if(is.null(data.valid)==F){
        gf.oos=gf(method,model,data.valid[,depvar],as.matrix(data.valid[,regvars]))
      }else{
        gf.oos=NULL
      }
    }
    
  }
  
  return(list(model,gf.ins,gf.oos))
}

estimate.and.test=function(method,depvar,gf.0,regvars.1,data.train,data.valid){
  
  results=estimate(method,depvar,regvars.1,data.train,data.valid) 
  model.1=results[[1]]
  gf.1=results[[2]]

  # LR-test
  n=nrow(data)
  LR.1=n*(log(gf.0["SSE"])-log(gf.1["SSE"]))
  names(LR.1)=paste0("LR.1.",method)
  LR.1.p.val=1-pchisq(LR.1,1)
  names(LR.1.p.val)=paste0("LR.1.p.val.",method)

  return(list(LR.1,LR.1.p.val))
  
}

# fit baseline model.0 (regvars.0) and large model.2 which includes 'testvar' in regvars.2

# linear model ####
method="lm"
results=estimate(method,depvar,regvars.0,data,NULL) # the model without the variable to be tested
model.0.lm=results[[1]]
gf.0.lm=results[[2]]

results=estimate(method,depvar,regvars.2,data,NULL) # the model including the variable to be tested
model.2.lm=results[[1]]
gf.2.lm=results[[2]]

# gbm ####
method="gbm"
results=estimate(method,depvar,regvars.0,data,NULL) # the model without the variable to be tested
model.0.gbm=results[[1]]
gf.0.gbm=results[[2]]

results=estimate(method,depvar,regvars.2,data,NULL) # the model including the variable to be tested
model.2.gbm=results[[1]]
gf.2.gbm=results[[2]]

# xgb ####
method="xgb"
results=estimate(method,depvar,regvars.0,data,NULL) # the model without the variable to be tested
model.0.xgb=results[[1]]
gf.0.xgb=results[[2]]

results=estimate(method,depvar,regvars.2,data,NULL) # the model including the variable to be tested
model.2.xgb=results[[1]]
gf.2.xgb=results[[2]]

# simulate LR distribution using all data ####
# in each repetition a model is fitted to the larger model in which testvar is replaced by 'testvar.rand'
# 'testvar.rand' is identical to 'testvar' but randomly sorted
# for each repetition a LR statistic is computed, comparing the model using 'testvar.rand' to the one without 'testvar'
# thereby we obtain a sampling distribution of the LR test-statistic under the null that 'testvar' is irrelevant

n.repeats=1000
LR.df=data.frame(array(NA,c(n.repeats,3)))
i=0
for(repeats in 1:n.repeats){
  
  # re-sort 'testvar' to create a random variable with the same distribution as 'testvar'
  u=runif(nrow(data))
  ui=sort(u,index.return=T)
  data[,"testvar.rand"]=data[ui$ix,testvar]
  
  results.lm=estimate.and.test("lm",depvar,gf.0.lm,regvars.1,data,NULL)
  i=i+1
  LR.df[i,]=cbind("lm",t(results.lm))

  results.gbm=estimate.and.test("gbm",depvar,gf.0.gbm,regvars.1,data,NULL)
  i=i+1
  LR.df[i,]=cbind("gbm",t(results.gbm))
  
  results.xgb=estimate.and.test("xgb",depvar,gf.0.xgb,regvars.1,data,NULL)
  i=i+1
  LR.df[i,]=cbind("xgb",t(results.xgb))
  
}

names(LR.df)=c("method","LR","LR.p.value")

# plot sampling distributions of the LR-stastic under the null that 'testvar' is irrelevant
ggplot(LR.df, aes(LR, fill=method)) + geom_density(alpha=0.4)

LR.lm=subset(LR.df,method=="lm")
LR.gbm=subset(LR.df,method=="gbm")
LR.xgb=subset(LR.df,method=="xgb")
LR.gbm.xgb=subset(LR.df,(method=="gbm")|(method=="xgb"))

#ggplot(LR.lm, aes(LR, fill=method)) + geom_density(alpha=0.4) + xlab("Sampling distribution of LR.1 assuming that 'age' is irrelevant")
ggplot(LR.lm, aes(LR, fill=method)) + geom_density(alpha=0.4) + xlab("LR.1")
ggplot(LR.gbm, aes(LR, fill=method)) + geom_density(alpha=0.4) + xlab("LR.1")
ggplot(LR.xgb, aes(LR, fill=method)) + geom_density(alpha=0.4) + xlab("LR.1")
ggplot(LR.gbm.xgb, aes(LR, fill=method)) + geom_density(alpha=0.4) + xlab("LR.1")

plot(LR.gbm$LR,LR.xgb$LR)
cor(LR.gbm$LR,LR.xgb$LR)

# LR tests based on simulated LR-distribution 
LR=function(method,data,LR.sim){
  
  assign("gf.0",paste0("gf.0.",method))
  assign("gf.2",paste0("gf.2.",method))
  
  n=nrow(data)
  LR.2=n*(log(get(gf.0)["SSE"])-log(get(gf.2)["SSE"]))
  names(LR.2)=paste0("LR.2.",method)
  LR.2.pval=1-pchisq(LR.2,1)
  names(LR.2.pval)=paste0("LR.2.",method,".pval")
  
  cat("LR.2 test statistic",method,":",LR.2,"\n")
  cat("p-value",method,":",LR.2.pval,"\n")
  cat("simulated p-value:",sum(LR.2<LR.sim["LR"])/nrow(LR.sim))

  return(list(LR.2,LR.2.pval))
  
}

LR.2=LR("lm",data,LR.lm)
LR.2=LR("gbm",data,LR.gbm)
LR.2=LR("xgb",data,LR.xgb)

# simulate in-and out-of-sample, cv-based LR-distributions ####

estimate.and.test.cv=function(method,depvar,regvars.0,regvars.1,regvars.2,data.train,data.valid){
  
  LR.df=data.frame(array(NA,c(4,5)))
  
  results=estimate(method,depvar,regvars.0,data.train,data.valid)
  gf.ins.0=results[[2]]
  gf.oos.0=results[[3]]
  
  results=estimate(method,depvar,regvars.1,data.train,data.valid)
  gf.ins.1=results[[2]]
  gf.oos.1=results[[3]]
  
  results=estimate(method,depvar,regvars.2,data.train,data.valid)
  gf.ins.2=results[[2]]
  gf.oos.2=results[[3]]
  
  # LR-test
  n=nrow(data[-fi,])
  LR.ins.1=n*(log(gf.ins.0["SSE"])-log(gf.ins.1["SSE"]))
  LR.ins.2=n*(log(gf.ins.0["SSE"])-log(gf.ins.2["SSE"]))
  
  n=nrow(data[fi,])
  LR.oos.1=n*(log(gf.oos.0["SSE"])-log(gf.oos.1["SSE"]))
  LR.oos.2=n*(log(gf.oos.0["SSE"])-log(gf.oos.2["SSE"]))

  LR.df[1,]=cbind(n.folds,method,"ins",1,LR.ins.1)
  LR.df[2,]=cbind(n.folds,method,"ins",2,LR.ins.2)
  LR.df[3,]=cbind(n.folds,method,"oos",1,LR.oos.1)
  LR.df[4,]=cbind(n.folds,method,"oos",2,LR.oos.2)
  
  return(LR.df)
  
}

fold.seq=seq(4,10,2)
n.repeats=1000
LR.cv.df=data.frame(array(NA,c(n.repeats,5))) # number of rows does not need to be correct! 
i=1
for(n.folds in fold.seq){
  
  # re-sort 'testvar' to create a random variable with the same distribution as 'testvar'
  u=runif(nrow(data))
  ui=sort(u,index.return=T)
  data[,"testvar.rand"]=data[ui$ix,testvar]
  
  for(repeats in 1:(floor(n.repeats/n.folds))){
    
    folds <- createFolds(data[,depvar],n.folds)
    
    for(k in 1:length(folds)){
      
      fi=unlist(folds[k]) # fi ... indices of test data
      
      LR.cv.lm=estimate.and.test.cv("lm",depvar,regvars.0,regvars.1,regvars.2,data[-fi,],data[fi,])
      LR.cv.gbm=estimate.and.test.cv("gbm",depvar,regvars.0,regvars.1,regvars.2,data[-fi,],data[fi,])
      LR.cv.xgb=estimate.and.test.cv("xgb",depvar,regvars.0,regvars.1,regvars.2,data[-fi,],data[fi,])
      
      LR.cv.df[i:(i+11),]=rbind(LR.cv.lm,LR.cv.gbm,LR.cv.xgb)
      i=i+12

    }
  }
}

names(LR.cv.df)=c("n.folds","method","io","eqn","LR")
LR.cv.df[,"LR"]=as.numeric(LR.cv.df[,"LR"])

ggplot(subset(LR.cv.df,(io=="ins")&(method=="lm")&(eqn==1)), aes(LR, fill=n.folds, linetype=n.folds)) + 
  geom_density(alpha=0.25) + xlab("LR.1 in-sample (linear model)")
ggplot(subset(LR.cv.df,(io=="ins")&((method=="xgb")|(method=="gbm"))&(eqn==1)), aes(LR, col=n.folds, linetype=n.folds, fill=method)) +
  geom_density(alpha=0.1) + xlab("LR.1 in-sample")

ggplot(subset(LR.cv.df,(io=="oos")&(method=="lm")&(eqn==1)), aes(LR, fill=n.folds, linetype=n.folds)) + 
  geom_density(alpha=0.25) + xlab("LR.1 out-of-sample (linear model)")
ggplot(subset(LR.cv.df,(io=="oos")&((method=="xgb")|(method=="gbm"))&(eqn==1)), aes(LR, col=n.folds, linetype=n.folds, fill=method)) +
  geom_density(alpha=0.1) + xlab("LR.1 out-of-sample")

ggplot(subset(LR.cv.df,(io=="oos")&(method=="lm")&(eqn==2)), aes(LR, fill=n.folds, linetype=n.folds)) + 
  geom_density(alpha=0.25) + xlab("LR.2 out-of-sample (linear model)")
ggplot(subset(LR.cv.df,(io=="oos")&((method=="xgb")|(method=="gbm"))&(eqn==2)), aes(LR, col=n.folds, linetype=n.folds, fill=method)) +
  geom_density(alpha=0.1) + xlab("LR.2 out-of-sample")

LR.cv=function(method_,io_,LR.df){

  LR1=subset(LR.df,(method==method_ & io==io_ & eqn=="1"))
  LR2=subset(LR.df,(method==method_ & io==io_ & eqn=="2")) 
  LR=subset(LR.df,(method==method_ & io==io_)) 
  names(LR)[4]="LR"
  names(LR)[5]="LR12"
  
  LR.cor=cor(LR2["LR"],LR1["LR"])
  cat("correlation between LR.1 and LR.2:",LR.cor,"\n")
  LR.pval=sum(LR2["LR"]<LR1["LR"])/nrow(LR1)
  cat("simulated p-value:",LR.pval)
  
  ggp=ggplot(LR, aes(LR12, fill=LR, linetype=n.folds)) + 
          geom_density(alpha=0.1) + xlab(paste0("LR.1 and LR.2 ",method_))
  
  print(ggp)
  
}

LR.cv("lm","oos",LR.cv.df)
LR.cv("gbm","oos",LR.cv.df)
LR.cv("xgb","oos",LR.cv.df)

# apply hdi ####
outLasso=lasso.proj(x = as.matrix(data[,regvars.2]),y=data[,depvar]) 
print(outLasso$bhat["age"])
print(outLasso$pval["age"])

outRidge=ridge.proj(x = as.matrix(data[,regvars.2]),y=data[,depvar]) 
print(outRidge$bhat["age"])
print(outRidge$pval["age"])
