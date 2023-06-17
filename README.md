## Overview

In this project, we are going to employ machine learning algorithms in
order to predict the manner in which the exercise was done based on the
recorded data.

## Data preprocessing

First we need to download and separate the data in training and testing
sets. Note that the test set is not the test set used during the test
and validation of the training. We also clean up the data.

``` r
data.trainingfile.url <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
data.testfile.url <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'

training.file <- './pml-training.csv'
test.file <- './pml-testing.csv'

download.file(data.trainingfile.url, training.file)
download.file(data.testfile.url, test.file)

training <- read.csv(training.file, na.strings = c("NA", "#DIV/0!" , ""), header = TRUE)
testing <- read.csv(test.file, na.strings = c("NA", "#DIV/0!" , ""), header = TRUE)

training<-training[,colSums(is.na(training))==0]
testing<-testing[,colSums(is.na(testing))==0]
training <- training[,-c(1:7)]
testing <- testing[,-c(1:7)]
```

## Cross validation

We partition the training dataset. We will then use one part of the data
(the testing set) to vadidate the models trained with the training part
of the data (the training set).

``` r
trainingset_i <- createDataPartition(training$classe, p = 0.75, list = FALSE)
trainingset <- training[trainingset_i, ]
testingset <- training[-trainingset_i, ]

dim(trainingset)
```

    ## [1] 14718    53

``` r
dim(testingset)
```

    ## [1] 4904   53

``` r
trainingset$classe <- as.factor(trainingset$classe)
testingset$classe <- as.factor(testingset$classe)
```

## Decision tree model

Next we will train a decision tree model

``` r
set.seed(12345)

DecisionTree <- rpart(classe ~ ., data=trainingset, method = "class")
predictionDecisionTree <- predict(DecisionTree, testingset, type ="class")

confusionMatrix(predictionDecisionTree, testingset$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1273  175   55   53   62
    ##          B   34  551   43   42   68
    ##          C   25  100  581   39   42
    ##          D   42   41  153  621   83
    ##          E   21   82   23   49  646
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7488          
    ##                  95% CI : (0.7364, 0.7609)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6807          
    ##                                           
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9125   0.5806   0.6795   0.7724   0.7170
    ## Specificity            0.9017   0.9527   0.9491   0.9222   0.9563
    ## Pos Pred Value         0.7868   0.7466   0.7382   0.6606   0.7868
    ## Neg Pred Value         0.9629   0.9045   0.9334   0.9538   0.9375
    ## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
    ## Detection Rate         0.2596   0.1124   0.1185   0.1266   0.1317
    ## Detection Prevalence   0.3299   0.1505   0.1605   0.1917   0.1674
    ## Balanced Accuracy      0.9071   0.7667   0.8143   0.8473   0.8366

## Random Forest model

Furthermore we will train a random forest model

``` r
ranForestModel <- randomForest(classe ~ ., data=trainingset, method = "class")
predictionForest <- predict(ranForestModel, testingset, type ="class")

confusionMatrix(predictionForest, testingset$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1394    3    0    0    0
    ##          B    1  942    4    0    0
    ##          C    0    4  850    9    0
    ##          D    0    0    1  794    1
    ##          E    0    0    0    1  900
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9951          
    ##                  95% CI : (0.9927, 0.9969)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9938          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9993   0.9926   0.9942   0.9876   0.9989
    ## Specificity            0.9991   0.9987   0.9968   0.9995   0.9998
    ## Pos Pred Value         0.9979   0.9947   0.9849   0.9975   0.9989
    ## Neg Pred Value         0.9997   0.9982   0.9988   0.9976   0.9998
    ## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
    ## Detection Rate         0.2843   0.1921   0.1733   0.1619   0.1835
    ## Detection Prevalence   0.2849   0.1931   0.1760   0.1623   0.1837
    ## Balanced Accuracy      0.9992   0.9957   0.9955   0.9935   0.9993

The accuracy of the random forest model (99,73%) is clearly higher than
the decision tree model (74,65%). The out-of-sample error is thus lower
for the random forest model (0,27%) than for the decision tree model
(25,35%).

## Quiz - Test

The random forest model is the most accurate, so we will use this for
prediction of the quiz data set (included in the test set)

``` r
testPrediction <- predict(ranForestModel, newdata=testing)
data.frame(classe=testPrediction)
```

    ##    classe
    ## 1       B
    ## 2       A
    ## 3       B
    ## 4       A
    ## 5       A
    ## 6       E
    ## 7       D
    ## 8       B
    ## 9       A
    ## 10      A
    ## 11      B
    ## 12      C
    ## 13      B
    ## 14      A
    ## 15      E
    ## 16      E
    ## 17      A
    ## 18      B
    ## 19      B
    ## 20      B
