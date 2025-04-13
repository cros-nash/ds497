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
mean(forScoresA$meanConfidence)
sd(forScoresA$meanConfidence)

forScoresB <- fullPerPpantDF %>% filter(intervention=="B")
mean(forScoresB$meanConfidence)
sd(forScoresB$meanConfidence)

forScoresC <- fullPerPpantDF %>% filter(intervention=="C")
mean(forScoresC$meanConfidence)
sd(forScoresC$meanConfidence)

forScoresD <- fullPerPpantDF %>% filter(intervention=="D")
mean(forScoresD$meanConfidence)
sd(forScoresD$meanConfidence)

mean(fullPerPpantDF$meanConfidence)
sd(fullPerPpantDF$meanConfidence)


mean(forScoresA$justLabelDFMeanConf,na.rm=TRUE)
sd(forScoresA$justLabelDFMeanConf,na.rm=TRUE)

mean(forScoresB$justLabelDFMeanConf,na.rm=TRUE)
sd(forScoresB$justLabelDFMeanConf,na.rm=TRUE)

mean(forScoresC$justLabelDFMeanConf,na.rm=TRUE)
sd(forScoresC$justLabelDFMeanConf,na.rm=TRUE)

mean(forScoresD$justLabelDFMeanConf,na.rm=TRUE)
sd(forScoresD$justLabelDFMeanConf,na.rm=TRUE)

mean(fullPerPpantDF$justLabelDFMeanConf,na.rm=TRUE)
sd(fullPerPpantDF$justLabelDFMeanConf,na.rm=TRUE)


mean(forScoresA$justLabelRealMeanConf,na.rm=TRUE)
sd(forScoresA$justLabelRealMeanConf,na.rm=TRUE)

mean(forScoresB$justLabelRealMeanConf,na.rm=TRUE)
sd(forScoresB$justLabelRealMeanConf,na.rm=TRUE)

mean(forScoresC$justLabelRealMeanConf,na.rm=TRUE)
sd(forScoresC$justLabelRealMeanConf,na.rm=TRUE)

mean(forScoresD$justLabelRealMeanConf,na.rm=TRUE)
sd(forScoresD$justLabelRealMeanConf,na.rm=TRUE)

mean(fullPerPpantDF$justLabelRealMeanConf,na.rm=TRUE)
sd(fullPerPpantDF$justLabelRealMeanConf,na.rm=TRUE)


mean(forScoresA$justTruthDFMeanConf,na.rm=TRUE)
sd(forScoresA$justTruthDFMeanConf,na.rm=TRUE)

mean(forScoresB$justTruthDFMeanConf,na.rm=TRUE)
sd(forScoresB$justTruthDFMeanConf,na.rm=TRUE)

mean(forScoresC$justTruthDFMeanConf,na.rm=TRUE)
sd(forScoresC$justTruthDFMeanConf,na.rm=TRUE)

mean(forScoresD$justTruthDFMeanConf,na.rm=TRUE)
sd(forScoresD$justTruthDFMeanConf,na.rm=TRUE)

mean(fullPerPpantDF$justTruthDFMeanConf,na.rm=TRUE)
sd(fullPerPpantDF$justTruthDFMeanConf,na.rm=TRUE)


mean(forScoresA$justTruthRealMeanConf,na.rm=TRUE)
sd(forScoresA$justTruthRealMeanConf,na.rm=TRUE)

mean(forScoresB$justTruthRealMeanConf,na.rm=TRUE)
sd(forScoresB$justTruthRealMeanConf,na.rm=TRUE)

mean(forScoresC$justTruthRealMeanConf,na.rm=TRUE)
sd(forScoresC$justTruthRealMeanConf,na.rm=TRUE)

mean(forScoresD$justTruthRealMeanConf,na.rm=TRUE)
sd(forScoresD$justTruthRealMeanConf,na.rm=TRUE)

mean(fullPerPpantDF$justTruthRealMeanConf,na.rm=TRUE)
sd(fullPerPpantDF$justTruthRealMeanConf,na.rm=TRUE)


anovScores <- aov(meanConfidence ~ intervention, data=fullPerPpantDF)
summary(anovScores)

aovLabelReal <- aov(justLabelRealMeanConf~intervention,data=fullPerPpantDF)
summary(aovLabelReal)

aovLabelDF <- aov(justLabelDFMeanConf~intervention,data=fullPerPpantDF)
summary(aovLabelDF)

aovTruthReal <- aov(justTruthRealMeanConf~intervention,data=fullPerPpantDF)
summary(aovTruthReal)

aovTruthDF <- aov(justTruthDFMeanConf~intervention,data=fullPerPpantDF)
summary(aovTruthDF)


ab <- t.test(forScoresA$justLabelDFMeanConf,forScoresB$justLabelDFMeanConf,var.equal=FALSE)
ab

ac <- t.test(forScoresA$justLabelDFMeanConf,forScoresC$justLabelDFMeanConf,var.equal=FALSE)
ac

ad <- t.test(forScoresA$justLabelDFMeanConf,forScoresD$justLabelDFMeanConf,var.equal=FALSE)
ad

bc <- t.test(forScoresB$justLabelDFMeanConf,forScoresC$justLabelDFMeanConf,var.equal=FALSE)
bc

bd <- t.test(forScoresB$justLabelDFMeanConf,forScoresD$justLabelDFMeanConf,var.equal=FALSE)
bd

cd <- t.test(forScoresC$justLabelDFMeanConf,forScoresD$justLabelDFMeanConf,var.equal=FALSE)
cd


ab <- t.test(forScoresA$justLabelRealMeanConf,forScoresB$justLabelRealMeanConf,var.equal=FALSE)
ab

ac <- t.test(forScoresA$justLabelRealMeanConf,forScoresC$justLabelRealMeanConf,var.equal=FALSE)
ac

ad <- t.test(forScoresA$justLabelRealMeanConf,forScoresD$justLabelRealMeanConf,var.equal=FALSE)
ad

bc <- t.test(forScoresB$justLabelRealMeanConf,forScoresC$justLabelRealMeanConf,var.equal=FALSE)
bc

bd <- t.test(forScoresB$justLabelRealMeanConf,forScoresD$justLabelRealMeanConf,var.equal=FALSE)
bd

cd <- t.test(forScoresC$justLabelRealMeanConf,forScoresD$justLabelRealMeanConf,var.equal=FALSE)
cd



ab <- t.test(forScoresA$justTruthDFMeanConf,forScoresB$justTruthDFMeanConf,var.equal=FALSE)
ab

ac <- t.test(forScoresA$justTruthDFMeanConf,forScoresC$justTruthDFMeanConf,var.equal=FALSE)
ac

ad <- t.test(forScoresA$justTruthDFMeanConf,forScoresD$justTruthDFMeanConf,var.equal=FALSE)
ad

bc <- t.test(forScoresB$justTruthDFMeanConf,forScoresC$justTruthDFMeanConf,var.equal=FALSE)
bc

bd <- t.test(forScoresB$justTruthDFMeanConf,forScoresD$justTruthDFMeanConf,var.equal=FALSE)
bd

cd <- t.test(forScoresC$justTruthDFMeanConf,forScoresD$justTruthDFMeanConf,var.equal=FALSE)
cd


ab <- t.test(forScoresA$justTruthRealMeanConf,forScoresB$justTruthRealMeanConf,var.equal=FALSE)
ab

ac <- t.test(forScoresA$justTruthRealMeanConf,forScoresC$justTruthRealMeanConf,var.equal=FALSE)
ac

ad <- t.test(forScoresA$justTruthRealMeanConf,forScoresD$justTruthRealMeanConf,var.equal=FALSE)
ad

bc <- t.test(forScoresB$justTruthRealMeanConf,forScoresC$justTruthRealMeanConf,var.equal=FALSE)
bc

bd <- t.test(forScoresB$justTruthRealMeanConf,forScoresD$justTruthRealMeanConf,var.equal=FALSE)
bd

cd <- t.test(forScoresC$justTruthRealMeanConf,forScoresD$justTruthRealMeanConf,var.equal=FALSE)
cd

