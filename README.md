# Click-Fraud-Classification
I will be covering a crucial topic in advertising, the click fraud. I will use a popular machine learning method to detect fraudulent behavior in practice: Decision Tree Models.

Let’s start!

Click Fraud in PPC Advertising
Digital advertising has been one of the most important channels in many industries within the last decade. In 2018, digital advertising spending worldwide has been 269.85 billion US Dollars (Statista, 2018) surpassing TV advertising spending. This growth, unfortunately, attracted many fraudulent actions in various digital advertising channels.
<img width="1000" height="743" alt="image" src="https://github.com/user-attachments/assets/7d675eef-12a9-4fc3-aeb6-18a5e6d2753d" />


In pay-per-click (PPC) advertising, some publishers create fraudulent clicks using various fraud techniques (bots, click networks, etc.) and make money through these illegitimate clicks. Multiple players, such as the publisher of the advertisement, stakeholders of the advertising network or even non-contracting parties (competitors, friends, etc.) may be the source of click frauds.

Advertisers are readily convinced that digital advertising and in particular the PPC ads are important channels that would raise consumers’ awareness. On the other hand, practices such as click fraud pull up the cost of (or diminish the returns on) the PPC ads substantially as a fraction of the clicks are not real. In 2017, on the average, half of the desktop impressions were fraudulent worldwide. According to Statista, in 2018 the cost of digital ad fraud has been 19 billion dollars. While various sectors are the target of these fraudulent actions, education, e-commerce, medical, legal and travel services are among the most affected sectors in 2018. Therefore, advertisers from various sectors respond to the click frauds with litigation (such as AdTrader versus Google case in California in 2017) or more proactively with ‘’click fraud protection’’.
<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/621c5a88-cacd-4219-a165-48bbda8643b4" />


In this project, I will be focusing on click fraud protection. I will implement a machine learning model through which one can understand which clicks are fraudulent and which are real (likely to turn into conversion). It is important to note that in practice various other methods and algorithms are being developed to detect the click frauds. TalkingData’s Click Fraud challenge published in Kaggle is a recent example of various protection practices: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection

Specifically, we will execute the decision tree models, including random forest model. Using the training data, we seek to understand which factors are suitable to predict if click origins from a real consumer or a fraudulent operator.

Before introducing the methods, let’s familiarizw ourselves with the data.

# Data
We will be covering the case of an app advertisement. We will focus on the data of ad clicks for an app and whether or not the app was downloaded after the click conversion. And we aim to predict whether a click is fraudulent or not.

Our dataset consists of two subsets: training set and testing set. It is important to note that for supervised learning methods (such as decision trees), the algorithm must be trained (supervised) with the training set to predict the outcomes of the test set.

Now, let’s start with loading the training and testing sets into the R environment. The datasets are available on our learning platform (“train_sample.csv”” and “test_sample.csv”“). First, you will need to download these datasets into a directory you work with. Next, you can load datasets to R as follows.

```r
data<-read.csv('train_sample.csv')

## 75% of the sample size
smp_size <- floor(0.75 * nrow(data))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(data)), size = smp_size)

train <- data[train_ind, ]
testpure <- data[-train_ind,]
test <- data[-train_ind,]
test$is_attributed <- NULL

test$click_id<-seq.int(nrow(test))
```
Now, if you check the environment tab of the upper right-hand side, you will see the datasets created. You can double-click on the datasets and observe their elements. In this example, the rows indicate the clicks and columns show relevant click features such as IP, device or the time stamp showing if the app was downloaded.

You can use the str() function to see the data types, the number of observations and the variables. You can also use other functions such as head() or summary() to explore the dataset. Let us try these functions:

```r
#Summarize the data
head(train)
```
```text
##           ip app device os channel          click_time attributed_time
## 51663  29367  12      1 41     481 2017-11-08 12:59:17                
## 57870   5077  15      1 19     265 2017-11-08 16:49:41                
## 2986  324569   3      1 22     280 2017-11-08 16:50:07                
## 29925 110119  12      1 22     497 2017-11-07 03:55:15                
## 95246  18183  64      1 31     459 2017-11-08 11:19:17                
## 68293  37549  12      1 19     340 2017-11-08 15:38:12                
##       is_attributed
## 51663             0
## 57870             0
## 2986              0
## 29925             0
## 95246             0
## 68293             0
```
```r
str(train)
```
```text
## 'data.frame':    75000 obs. of  8 variables:
##  $ ip             : int  29367 5077 324569 110119 18183 37549 84758 204196 132823 80908 ...
##  $ app            : int  12 15 3 12 64 12 1 26 9 3 ...
##  $ device         : int  1 1 1 1 1 1 1 1 1 1 ...
##  $ os             : int  41 19 22 22 31 19 13 20 13 22 ...
##  $ channel        : int  481 265 280 497 459 340 134 121 489 173 ...
##  $ click_time     : chr  "2017-11-08 12:59:17" "2017-11-08 16:49:41" "2017-11-08 16:50:07" "2017-11-07 03:55:15" ...
##  $ attributed_time: chr  "" "" "" "" ...
##  $ is_attributed  : int  0 0 0 0 0 0 0 0 0 0 ...
```
```r
#One can also use summary() function to see summary information associated with each variable
summary(train)
```
```text
##        ip              app             device              os        
##  Min.   :     9   Min.   :  1.00   Min.   :   0.00   Min.   :  0.00  
##  1st Qu.: 40631   1st Qu.:  3.00   1st Qu.:   1.00   1st Qu.: 13.00  
##  Median : 79856   Median : 12.00   Median :   1.00   Median : 18.00  
##  Mean   : 91228   Mean   : 12.07   Mean   :  21.66   Mean   : 22.82  
##  3rd Qu.:118248   3rd Qu.: 15.00   3rd Qu.:   1.00   3rd Qu.: 19.00  
##  Max.   :364648   Max.   :551.00   Max.   :3867.00   Max.   :866.00  
##     channel       click_time        attributed_time    is_attributed    
##  Min.   :  3.0   Length:75000       Length:75000       Min.   :0.00000  
##  1st Qu.:145.0   Class :character   Class :character   1st Qu.:0.00000  
##  Median :258.0   Mode  :character   Mode  :character   Median :0.00000  
##  Mean   :268.9                                         Mean   :0.00236  
##  3rd Qu.:379.0                                         3rd Qu.:0.00000  
##  Max.   :498.0                                         Max.   :1.00000
```
How many variables and observations do we have in the training data set?

In particular, the variables representing the click features are the IP addresses (ip), the app id (app), the device type (device), the operating system version (os), the channel id of mobile add publisher (channel), the time stamp of the click (click_time) and the time of app download (the attributed_time). Note that the last column (is_attributed) denotes whether or not a click is fraudulent. If it is 0 (as for the first row), then the associated click is fraudulent.If it is 1 (as in row 482), then it is a real click. What do you observe about the relationship between (is_attributed) and the (the attributed_time)?

```r
#Training Set
data<-read.csv('train_sample.csv')

## 75% of the sample size
smp_size <- floor(0.75 * nrow(data))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(data)), size = smp_size)

train <- data[train_ind, ]
testpure <- data[-train_ind,]
test <- data[-train_ind,]
test$is_attributed <- NULL

test$click_id<-seq.int(nrow(test))
head(test)
```
```text
##        ip app device os channel          click_time attributed_time click_id
## 2  105560  25      1 17     259 2017-11-07 13:40:27                        1
## 13 114809   3      1 22     205 2017-11-09 10:24:23                        2
## 14 114220   6      1 20     125 2017-11-08 14:46:16                        3
## 24   8362   7      1 19     101 2017-11-07 10:30:00                        4
## 26 145896  64      1 13     459 2017-11-07 03:58:58                        5
## 27 162976   3      1 13     115 2017-11-07 16:19:05                        6
```
```r
str(test)
```
```text
## 'data.frame':    25000 obs. of  8 variables:
##  $ ip             : int  105560 114809 114220 8362 145896 162976 52432 127888 87879 250933 ...
##  $ app            : int  25 3 6 7 64 3 1 13 3 3 ...
##  $ device         : int  1 1 1 1 1 1 1 1 1 1 ...
##  $ os             : int  17 22 20 19 13 13 13 23 13 13 ...
##  $ channel        : int  259 205 125 101 459 115 115 477 115 280 ...
##  $ click_time     : chr  "2017-11-07 13:40:27" "2017-11-09 10:24:23" "2017-11-08 14:46:16" "2017-11-07 10:30:00" ...
##  $ attributed_time: chr  "" "" "" "" ...
##  $ click_id       : int  1 2 3 4 5 6 7 8 9 10 ...
```
```r
#One can also use summary() function to see summary information associated with each variable
summary(test)
```
```text
##        ip              app             device              os        
##  Min.   :    10   Min.   :  1.00   Min.   :   0.00   Min.   :  0.00  
##  1st Qu.: 40433   1st Qu.:  3.00   1st Qu.:   1.00   1st Qu.: 13.00  
##  Median : 79710   Median : 12.00   Median :   1.00   Median : 18.00  
##  Mean   : 91338   Mean   : 11.98   Mean   :  22.12   Mean   : 22.82  
##  3rd Qu.:118315   3rd Qu.: 15.00   3rd Qu.:   1.00   3rd Qu.: 19.00  
##  Max.   :364757   Max.   :486.00   Max.   :3866.00   Max.   :866.00  
##     channel       click_time        attributed_time       click_id    
##  Min.   :  3.0   Length:25000       Length:25000       Min.   :    1  
##  1st Qu.:140.0   Class :character   Class :character   1st Qu.: 6251  
##  Median :258.0   Mode  :character   Mode  :character   Median :12500  
##  Mean   :268.7                                         Mean   :12500  
##  3rd Qu.:386.0                                         3rd Qu.:18750  
##  Max.   :497.0                                         Max.   :25000
```
We note that the test dataset is similar to the train set except that there exists an extra column (click_id) which is the reference for making predictions and (is_attributed) column is not included. Indeed, we are supposed to predict the (is_attributed) column with our machine learning algorithm.

Question: Which variables do you think are the most important to predict if a click is a fraudulent or not (except the attributed_time)?

```r
#install.packages("tidyr", repos="http://cran.us.r-project.org")
#library(tidyr)

#separate click time into date and time 
train=separate(train, col=click_time,into=c("date_click","time_click"), sep=" ")
test=separate(test, col=click_time,into=c("date_click","time_click"), sep=" ")
#further classify date_click to weekdays
#install.packages("date")
#install.packages("lubridate")
#library(date)
#library(lubridate)

train$date_click<-wday(as.Date(train$date_click))
train$time_click<-hms(train$time_click)
train$time_click<-period_to_seconds(train$time_click)
train$attributed_time<-NULL
train$is_attributed<-as.factor(train$is_attributed)
test$date_click<-wday(as.Date(test$date_click))
test$time_click<-hms(test$time_click)
test$time_click<-period_to_seconds(test$time_click)
test$attributed_time<-NULL
```
We have familiarized ourselves with the training data. The next step is to introduce the decision tree models.

# Decision Tree Models
Decision Tree Models are a form of “supervised” learning to solve various problems including the click fraud problem. Our question is whether a click is real or fraudulent. Therefore, we have a classification problem with a qualitative outcome. Note, however, that decision tree models can also be used for continuous settings with quantitative outcomes.

The decision tree method belongs to the supervised learning methods. Datasets for supervised learning methods consist of training and a testing data set because the algorithm must be trained (supervised) with one of the datasets to test and predict reliable results on the other one.

Conceptually, the decision tree algorithm starts with all the data at the root node and scans all the variables for the best one to split on. Once a variable is chosen, you do the split and go down one level (or one node) and repeat. The final nodes at the bottom of the decision tree are known as terminal nodes, and the majority vote of the observations in that node determine how to predict for new observations that end up in that terminal node.

To create your first decision tree, you’ll make use of R’s rpart package. Instead of writing an algorithm yourself, you can use this package to build your decision tree.

```r
#install.packages("rpart")
#install.packages("rattle")
library(rpart)

# Build the decision tree
#my_tree<- rpart(is_attributed ~ ip + app + device + date_click + time_click + os , data = train, method = "class")

my_tree<- rpart(is_attributed ~ ip + app  + date_click + time_click +os, data = train, method = "class")

# Visualize the decision tree using plot() and text()
plot(my_tree)
text(my_tree)
```
<img width="639" height="357" alt="截屏2026-02-03 上午11 08 15" src="https://github.com/user-attachments/assets/c982709a-0164-4165-8456-158b2c63dd70" />


As you see, the decision tree plot based on the rpart package is quite hard to read . Let us download the package named rattle and obtain the decision tree plots and interpret them.

```r
# Load in the packages to build a fancy plot
install.packages("rattle", repos="http://cran.us.r-project.org")
```
```text
## 
## The downloaded binary packages are in
##  /var/folders/gv/4k65csb96b1gt9gdz15_54kh0000gn/T//RtmpGq4IsS/downloaded_packages
```
```r
library(rattle)
```
```text
## Loading required package: tibble
```
```text
## Loading required package: bitops
```
```text
## Rattle: A free graphical interface for data science with R.
## Version 5.5.1 Copyright (c) 2006-2021 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```
```r
fancyRpartPlot(my_tree)
```
<img width="644" height="405" alt="截屏2026-02-03 上午11 09 59" src="https://github.com/user-attachments/assets/efcb2b7a-dade-426b-8173-ea00fe5e1845" />



This plot shows the occurrences of fraud click on various events. The “0” or “1” you get on the top of each node is determined by which number has the higher frequency (i.e, proportion of “0”s and “1”s). The percentage you get in the bottom of each node is the percentage of observations of your dataset (hence you always start from 100%).

As an example, for apps that thas id less than 29, the fradulent rate is 100%, and these constitute 97% of the data in the training set. If the app has an id higher than or equal to 29, we need to look at the lower order nodes. In this example, “ip” is another indicator that predicts whether or not a click is fraudulent or not. Looking at the node with higher probability of a genuine click (e.g., the one with 7% probability of fraudulent and 93% of genuine), the fradulent rate can be low for specific apps (e.g., app ID 34 and 35) that are in general not likely to be used.

We note that the colors of nodes change from green to blue as the classification becomes less narrow and approaches to sample. From a higher point of view, we can conclude that app ID, ip adresses and operatins systems are powerful traits that suggest whether a click is fraud or genuine.

Now let us test the prediction performance of the tree using the test set and then compare the predicted result with actual data in the test set.

```r
table(my_solution[,2],testpure$is_attributed)
```
```text
##    
##         0     1
##   0 24944    44
##   1     6     6
```
The rows refer to the actual data from the testing set and the columns refer to our predictions. We can interpret from the matrix that, of the 24950 fradulent clicks, our model is able to accurately predict 24944 (or 99.96%) of them. For the 50 genuine clicks, our model is able to accurately predict 6 (or 12%) of them. We can see that our model performs very well in detecting fraudulent clicks. The relatively lower accuracy with genuine clicks here can be due to the fact that our dataset has too few cases of genuine clicks (50 out of 25000 clicks), which is not very common in the real practice.

Now, let us try another type of algorithm within the family of the decision tree models, the random forest. The main idea in the random forest is to decorrelate the several trees which are generated by the different bootstrapped samples from training data. As the second step, the Variance is reduced by averaging several trees. Therefore, the performance and the precision improves. It is one of the most commong techniques among decision trees.

In R, random forest is implemented using the randomForest package.To apply the algorithm, we need to apply the randomForest() function. Now, let us implement the algorithm:

```r
# Load in the package
install.packages("randomForest", repos="http://cran.us.r-project.org")
```
```text
## 
## The downloaded binary packages are in
##  /var/folders/gv/4k65csb96b1gt9gdz15_54kh0000gn/T//RtmpGq4IsS/downloaded_packages
```
```r
library(randomForest)
```
```text
## randomForest 4.7-1.1
```
```text
## Type rfNews() to see new features/changes/bug fixes.
```
```text
## 
## Attaching package: 'randomForest'
```
```text
## The following object is masked from 'package:rattle':
## 
##     importance
```
```r
# Train set and test set
str(train)
```
```text
## 'data.frame':    75000 obs. of  8 variables:
##  $ ip           : int  29367 5077 324569 110119 18183 37549 84758 204196 132823 80908 ...
##  $ app          : int  12 15 3 12 64 12 1 26 9 3 ...
##  $ device       : int  1 1 1 1 1 1 1 1 1 1 ...
##  $ os           : int  41 19 22 22 31 19 13 20 13 22 ...
##  $ channel      : int  481 265 280 497 459 340 134 121 489 173 ...
##  $ date_click   : num  4 4 4 3 4 4 3 5 4 4 ...
##  $ time_click   : num  46757 60581 60607 14115 40757 ...
##  $ is_attributed: Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
```
```r
str(test)
```
```text
## 'data.frame':    25000 obs. of  8 variables:
##  $ ip        : int  105560 114809 114220 8362 145896 162976 52432 127888 87879 250933 ...
##  $ app       : int  25 3 6 7 64 3 1 13 3 3 ...
##  $ device    : int  1 1 1 1 1 1 1 1 1 1 ...
##  $ os        : int  17 22 20 19 13 13 13 23 13 13 ...
##  $ channel   : int  259 205 125 101 459 115 115 477 115 280 ...
##  $ date_click: num  3 5 4 3 3 3 3 2 3 4 ...
##  $ time_click: num  49227 37463 53176 37800 14338 ...
##  $ click_id  : int  1 2 3 4 5 6 7 8 9 10 ...
```
```r
# Set seed for reproducibility
set.seed(111)

# Apply the Random Forest Algorithm
memory.limit(size = 56000)
```
```text
## [1] Inf
```
```r
my_forest <- randomForest(as.factor(is_attributed) ~ ip + app + date_click + time_click + os,
                          data = train, importance = TRUE, ntree = 1000)

# Make your prediction using the test set
my_prediction <- predict(my_forest, test)

# Create a data frame with two columns: click_id & is_attributed, is_attributed contains your predictions
# Finish the data.frame() call
my_solutionrf <- data.frame(click_id = test$click_id, is_attributed = my_prediction, real=testpure$is_attributed)

# Use nrow() on my_solution
nrow(my_solutionrf)
```
```text
## [1] 25000
```
```r
# Finish the write.csv() call
write.csv(my_solutionrf, file = "my_solutionrf.csv", row.names = FALSE)
```
Let us know, compare our prediction of test data versus the actual values:

```r
table(my_solutionrf[,2],testpure$is_attributed)
```
```text
##    
##         0     1
##   0 24946    46
##   1     4     4
```
Here, similarly with our interpretation earlier, the random forest model can accurately capture most of the fraudulent clicks (24946 out of 24950), while performs relatively weaker in detecting genuine clicks (4 out of 50).

Next, we use varImPlot() function to visualize the variable importance as measured by a Random Forest:

```r
varImpPlot(my_forest)
```
<img width="703" height="464" alt="截屏2026-02-03 上午11 14 29" src="https://github.com/user-attachments/assets/89bc283e-0715-479c-aea0-14358b0dd4f9" />


When running the function, two graphs appear: the accuracy plot shows how much worse the model would perform without the included variables. So a big decrease (= high-value x-axis) links to a high predictive variable. The second plot is the Gini coefficient. The higher the variable scores here, the more important it is for the model.

Question: Based on the two plots, what variable has the highest impact on the model? 

Judging from the left panel, app id seems to be the most important variable since the model prediction accuracy would on average decrease by 40% without including it. The right panel informs us that app id could also lead to a relatively large decrease in the Gini coefficient (around 80).

IP address is also an important variable given that it has the highest score on the right panel, yet it is not as critical as app id in terms of prediction accuracy.

Note that you can improve the model using different model specifications (e.g. you can exclude some of the independent variables).
