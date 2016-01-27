library("caret")

set.seed(1134)
iris_trainIndex <- createDataPartition(iris$Species, p=.8,
                                  list=FALSE,
                                  times=1)
head(iris_trainIndex)
irisTrain <- iris[ iris_trainIndex, ]
irisTest <- iris[-iris_trainIndex, ]

LifeExpectancy = read.csv("life_expectancy.csv")
LifeExpectancy$Country <- NULL

set.seed(1134)
life_trainIndex <- createDataPartition(LifeExpectancy$Continent, p=.8,
                                  list=FALSE,
                                  times=1)
head(life_trainIndex)
LifeTrain <- LifeExpectancy[ life_trainIndex, ]
LifeTest <- LifeExpectancy[-life_trainIndex, ]


#ripper method for life expectancy
set.seed(1134)
control <- trainControl(method = "repeatedcv", repeats = 5, number = 10, verboseIter = T)

ripperMod <- train(Continent~ ., data = LifeTrain, method = "JRip",
                   trControl = control)
#ripper predict on test set
ripperPred <- predict(ripperMod, newdata= LifeTest)

summary(ripperPred)
confusionMatrix(ripperPred, reference = LifeTest$Continent)


#C4.5 life expectancy
set.seed(1134)
Cmod <- train(Continent~ ., data=LifeTrain, method= "J48", trControl = control)
summary(Cmod)

CPred <- predict(Cmod, newdata= LifeTest)
summary(CPred)
confusionMatrix(CPred, reference= LifeTest$Continent)


#oblique method for life expectancy 
set.seed(1134)
obliqueMod <- train(Continent~ ., data=LifeTrain, method ="oblique.tree",
                   trControl = control)

ObliquePred <- predict(obliqueMod, newdata= LifeTest)
summary(ObliquePred)
confusionMatrix(ObliquePred, reference = LifeTest$Continent)


#naive bayes for life expectancy
set.seed(1134)
naive <- train(Continent~ ., data=LifeTrain, method ="nb",
                    trControl = control)
naivePred <- predict(naive, newdata= LifeTest)
summary(naivePred)
confusionMatrix(naivePred, reference = LifeTest$Continent)
plot(naive)

#k-nearest neighbors for life expectancy
set.seed(1134)
knn <- train(Continent~ ., data=LifeTrain, method ="rknn",
               trControl = control)

knnPred <- predict(knn, newdata= LifeTest)
summary(knnPred)
confusionMatrix(knnPred, reference = LifeTest$Continent)
plot(knn)

#ripper method for iris
set.seed(1134)
ripperMod <- train(Species~ ., data = irisTrain, method = "JRip",
                   trControl = control)
#ripper predict on test set
ripperPred <- predict(ripperMod, newdata= irisTest)

summary(ripperPred)
confusionMatrix(ripperPred, reference = irisTest$Species)
plot(ripperMod)

#C4.5 iris
set.seed(1134)
Cmod <- train(Species~ ., data= irisTrain, method= "J48", trControl = control)

CPred <- predict(Cmod, newdata= irisTest)
summary(CPred)
confusionMatrix(CPred, reference= irisTest$Species)
plot(Cmod)

#oblique method foriris
set.seed(1134)
obliqueMod <- train(Species~ ., data=irisTrain, method ="oblique.tree",
                    trControl = control)

ObliquePred <- predict(obliqueMod, newdata= irisTest)
summary(ObliquePred)
confusionMatrix(ObliquePred, reference = irisTest$Species)
plot(obliqueMod)

#naive bayes for iris
set.seed(1134)
naive <- train(Species~ ., data=irisTrain, method ="nb",
               trControl = control)
naivePred <- predict(naive, newdata= irisTest)
summary(naivePred)
confusionMatrix(naivePred, reference = irisTest$Species)
plot(naive)

#k-nearest neighbors for iris
set.seed(1134)
knn <- train(Species~ ., data=irisTrain, method ="rknn",
             trControl = control)

knnPred <- predict(knn, newdata= irisTest)
summary(knnPred)
confusionMatrix(knnPred, reference = irisTest$Species)
plot(knn)