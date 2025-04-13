library(dplyr)
library(effsize)
library(ggplot2)
library(readr)


csv <- read_csv("eachresponse.csv")
csv <- csv %>% filter(!is.na(image))

ppantMeans <- csv
ppantMeans<- ppantMeans %>% mutate(correct=replace(correct,correct=="correct",1))
ppantMeans<- ppantMeans %>% mutate(correct=replace(correct,correct=="incorrect",0))
ppantMeans$correct <- as.numeric(ppantMeans$correct)

newdf <- aggregate(x=ppantMeans$confidence, by=list(ppantMeans$ppantID), FUN=mean)
names(newdf)<-c("ppantID","meanConfidence")
newdf$totalScore <- aggregate(x=ppantMeans$correct, by=list(ppantMeans$ppantID), FUN=sum)[,2]

summarizeDF <- ppantMeans %>% group_by(ppantID) %>% dplyr::summarize(intervention = last(intervention), n=n())

justLabelDF <- ppantMeans %>% filter(choice=='ai')
newdf2 <- aggregate(x=justLabelDF$confidence, by=list(justLabelDF$ppantID), FUN=mean)
names(newdf2)<-c("ppantID","justLabelDFMeanConf")
newdf2$justLabelDFMeanCorrect <- aggregate(x=justLabelDF$correct, by=list(justLabelDF$ppantID), FUN=mean)[,2]

justLabelReal <- ppantMeans %>% filter(choice=='real')
newdf3 <- aggregate(x=justLabelReal$confidence, by=list(justLabelReal$ppantID), FUN=mean)
names(newdf3)<-c("ppantID","justLabelRealMeanConf")
newdf3$justLabelRealMeanCorrect <- aggregate(x=justLabelReal$correct, by=list(justLabelReal$ppantID), FUN=mean)[,2]

justTruthDF <- ppantMeans %>% filter(choice=='real', correct==0) 
justTruthDF1 <- ppantMeans %>% filter(choice=='ai', correct==1)
justTruthDF <- rbind(justTruthDF,justTruthDF1)
newdf4 <- aggregate(x=justTruthDF$confidence, by=list(justTruthDF$ppantID), FUN=mean)
names(newdf4)<-c("ppantID","justTruthDFMeanConf")
newdf4$justTruthDFMeanCorrect <- aggregate(x=justTruthDF$correct, by=list(justTruthDF$ppantID), FUN=mean)[,2]

justTruthReal <- ppantMeans %>% filter(choice=='real', correct==1) 
justTruthReal1 <- ppantMeans %>% filter(choice=='ai', correct==0)
justTruthReal <- rbind(justTruthReal,justTruthReal1)
newdf5 <- aggregate(x=justTruthReal$confidence, by=list(justTruthReal$ppantID), FUN=mean)
names(newdf5)<-c("ppantID","justTruthRealMeanConf")
newdf5$justTruthRealMeanCorrect <- aggregate(x=justTruthReal$correct, by=list(justTruthReal$ppantID), FUN=mean)[,2]

perimgDF <- aggregate(x=ppantMeans$correct, by=list(ppantMeans$image), FUN=mean)
names(perimgDF)<-c("image","meanCorrectness")
perimgDF$meanConf <- aggregate(x=ppantMeans$confidence,by=list(ppantMeans$image),FUN=mean)[,2]

fullPerPpantDF<-left_join(newdf,summarizeDF,by='ppantID')
fullPerPpantDF <- left_join(fullPerPpantDF,newdf2,by='ppantID')
fullPerPpantDF <- left_join(fullPerPpantDF,newdf3,by='ppantID')
fullPerPpantDF <- left_join(fullPerPpantDF,newdf4,by='ppantID')
fullPerPpantDF <- left_join(fullPerPpantDF,newdf5,by='ppantID')

forScoresA <- fullPerPpantDF %>% filter(intervention=="A")
forScoresB <- fullPerPpantDF %>% filter(intervention=="B")
forScoresC <- fullPerPpantDF %>% filter(intervention=="C")
forScoresD <- fullPerPpantDF %>% filter(intervention=="D")

model <- lm(totalScore~meanConfidence, data = forScoresD)
model
summary(model)

cor.test(forScoresA$meanConfidence, forScoresA$totalScore, 
         method='pearson')
cor.test(forScoresB$meanConfidence, forScoresB$totalScore, 
         method='pearson')
cor.test(forScoresC$meanConfidence, forScoresC$totalScore, 
         method='pearson')
cor.test(forScoresD$meanConfidence, forScoresD$totalScore, 
         method='pearson')
