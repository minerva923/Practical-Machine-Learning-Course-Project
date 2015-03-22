# Practical Machine Learning Course Project
Minerva923  

##Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

##Data Processing
After loading needed packages and data, I observe the dataset first. I then decide to remove all variables having missing values since they tend to account most of that variable. When done, the data are splited to training and testing sets, waiting for further modeling and prediction.

```r
## Load needed packages
library(caret); library(parallel); library(doParallel)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
## Loading required package: foreach
## Loading required package: iterators
```

```r
registerDoParallel(clust <- makeForkCluster(detectCores()))
## Load training data
data <- read.csv("./pml-training.csv", na.strings = c("NA", "#DIV/0!", ""),
                 stringsAsFactors=FALSE)
## Remove all variables having missing value via observation
keep <- c("classe", "magnet_forearm_y", "magnet_forearm_z", "gyros_forearm_x",
          "gyros_forearm_y", "gyros_forearm_z", "accel_forearm_x",
          "accel_forearm_y", "accel_forearm_z", "magnet_forearm_x", 
          "total_accel_forearm", "accel_dumbbell_z", "magnet_dumbbell_x",
          "magnet_dumbbell_y", "magnet_dumbbell_z", "roll_forearm", 
          "pitch_forearm", "yaw_forearm", "gyros_dumbbell_x", "gyros_dumbbell_y",
          "gyros_dumbbell_z", "accel_dumbbell_x", "accel_dumbbell_y",
          "total_accel_dumbbell", "pitch_dumbbell", "yaw_dumbbell", "roll_dumbbell",
          "accel_arm_y", "accel_arm_z", "magnet_arm_x", "magnet_arm_y",
          "magnet_arm_z", "gyros_arm_x", "gyros_arm_y", "gyros_arm_z",
          "accel_arm_x", "magnet_belt_x", "magnet_belt_y", "magnet_belt_z", 
          "roll_arm", "pitch_arm", "yaw_arm", "total_accel_arm", "gyros_belt_x",
          "gyros_belt_y", "gyros_belt_z", "accel_belt_x", "accel_belt_y",
          "accel_belt_z", "roll_belt", "pitch_belt", "yaw_belt", "total_accel_belt")
data <- data[, keep]

set.seed(1234)
inTrain <- createDataPartition(y=data$classe, p=0.75, list=FALSE)
training <- data[inTrain,]
testing <- data[-inTrain,]
```

##Modeling
In the model, Random Forest algorithm is used, while the number of cross-validation is set 5 and pca is choosen to do the preprocess. In addition, I control the number of trees to be 50 through train command in order to reduce the computing time.

```r
fitControl <- trainControl(method = "cv", number = 5, allowParallel = TRUE)
modelFit <- train(factor(classe)~., data=training, method="rf", preProcess="pca",
                  trControl = fitControl, ntrees=50)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
## Turn off the paraell computing
stopCluster(clust)
```

##Prediction

```r
varImp(modelFit)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 25)
## 
##      Overall
## PC8   100.00
## PC12   88.31
## PC14   83.02
## PC1    82.97
## PC5    73.68
## PC3    69.75
## PC15   59.32
## PC9    58.32
## PC2    54.03
## PC6    52.14
## PC16   46.93
## PC22   45.92
## PC25   45.52
## PC21   44.98
## PC7    41.78
## PC17   41.37
## PC4    40.33
## PC10   40.26
## PC13   39.24
## PC11   31.21
```

```r
prediction <- predict(modelFit, newdata = testing)
print(cm <- confusionMatrix(prediction, testing$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1390   17    1    3    0
##          B    2  926   18    1    2
##          C    2    6  825   28    6
##          D    1    0    9  771    8
##          E    0    0    2    1  885
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9782          
##                  95% CI : (0.9737, 0.9821)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9724          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9964   0.9758   0.9649   0.9590   0.9822
## Specificity            0.9940   0.9942   0.9896   0.9956   0.9993
## Pos Pred Value         0.9851   0.9758   0.9516   0.9772   0.9966
## Neg Pred Value         0.9986   0.9942   0.9926   0.9920   0.9960
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2834   0.1888   0.1682   0.1572   0.1805
## Detection Prevalence   0.2877   0.1935   0.1768   0.1609   0.1811
## Balanced Accuracy      0.9952   0.9850   0.9773   0.9773   0.9907
```
Observing the result of variable importance, value of the first 7 variables are all larger than 60%. In the confusion matrix, we can see the accuracy, which in this case is 0.978, while Kappa value is 0.97. The result is satisfied.

The estimated out-of-sample error is 1 - the model accuracy

```r
ose <- 1 - cm$overall[1]
names(ose) <- "Out of Sample Error"
ose
```

```
## Out of Sample Error 
##          0.02181892
```


