Practical_ML_project
Panagiotis Paschalidis
17 06 2023
Overview
In this project, we are going to employ machine learning algorithms in order to predict the manner in which the exercise was done based on the recorded data.

Data preprocessing
First we need to download and separate the data in training and testing sets. Note that the test set is not the test set used during the test and validation of the training. We also clean up the data.

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
Cross validation
We partition the training dataset. We will then use one part of the data (the testing set) to vadidate the models trained with the training part of the data (the training set).

trainingset_i <- createDataPartition(training$classe, p = 0.75, list = FALSE)
trainingset <- training[trainingset_i, ]
testingset <- training[-trainingset_i, ]

dim(trainingset)
## [1] 14718    53
dim(testingset)
## [1] 4904   53
trainingset$classe <- as.factor(trainingset$classe)
testingset$classe <- as.factor(testingset$classe)
Decision tree model
Next we will train a decision tree model

set.seed(12345)

DecisionTree <- rpart(classe ~ ., data=trainingset, method = "class")
predictionDecisionTree <- predict(DecisionTree, testingset, type ="class")

confusionMatrix(predictionDecisionTree, testingset$classe)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1287  198   12   83   39
##          B   20  462   64   32   50
##          C   48  133  701  123  113
##          D   18   76   60  503   55
##          E   22   80   18   63  644
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7335          
##                  95% CI : (0.7209, 0.7458)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6613          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9226  0.48683   0.8199   0.6256   0.7148
## Specificity            0.9054  0.95803   0.8970   0.9490   0.9543
## Pos Pred Value         0.7949  0.73567   0.6270   0.7065   0.7787
## Neg Pred Value         0.9671  0.88611   0.9593   0.9282   0.9370
## Prevalence             0.2845  0.19352   0.1743   0.1639   0.1837
## Detection Rate         0.2624  0.09421   0.1429   0.1026   0.1313
## Detection Prevalence   0.3301  0.12806   0.2280   0.1452   0.1686
## Balanced Accuracy      0.9140  0.72243   0.8584   0.7873   0.8345
Random Forest model
Furthermore we will train a random forest model

ranForestModel <- randomForest(classe ~ ., data=trainingset, method = "class")
predictionForest <- predict(ranForestModel, testingset, type ="class")

confusionMatrix(predictionForest, testingset$classe)
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    1    0    0    0
##          B    1  946    5    0    0
##          C    0    2  849    7    1
##          D    0    0    1  796    3
##          E    0    0    0    1  897
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9955          
##                  95% CI : (0.9932, 0.9972)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9943          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9993   0.9968   0.9930   0.9900   0.9956
## Specificity            0.9997   0.9985   0.9975   0.9990   0.9998
## Pos Pred Value         0.9993   0.9937   0.9884   0.9950   0.9989
## Neg Pred Value         0.9997   0.9992   0.9985   0.9981   0.9990
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2843   0.1929   0.1731   0.1623   0.1829
## Detection Prevalence   0.2845   0.1941   0.1752   0.1631   0.1831
## Balanced Accuracy      0.9995   0.9977   0.9953   0.9945   0.9977
The accuracy of the random forest model (99,73%) is clearly higher than the decision tree model (74,65%). The out-of-sample error is thus lower for the random forest model (0,27%) than for the decision tree model (25,35%).

Quiz - Test
The random forest model is the most accurate, so we will use this for prediction of the quiz data set (included in the test set)

testPrediction <- predict(ranForestModel, newdata=testing)
data.frame(classe=testPrediction)
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
