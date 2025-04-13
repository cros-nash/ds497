library(dplyr)
library(effsize)
library(ggplot2)
library(plotly)
library(readr)
library("plotrix")


csv <- read_csv("eachresponse.csv")
csv <- csv %>% filter(!is.na(image))

ppantMeans <- csv
ppantMeans<- ppantMeans %>% mutate(correct=replace(correct,correct=="correct",1))
ppantMeans<- ppantMeans %>% mutate(correct=replace(correct,correct=="incorrect",0))
ppantMeans$correct <- as.numeric(ppantMeans$correct)

newdf <- aggregate(x=ppantMeans$confidence, by=list(ppantMeans$ppantID), FUN=mean)
names(newdf)<-c("ppantID","meanConfidence")
newdf$totalScore <- aggregate(x=ppantMeans$correct, by=list(ppantMeans$ppantID), FUN=sum)[,2]

summarizeDF <- ppantMeans %>% group_by(ppantID) %>% dplyr::summarize(intervention = last(intervention))

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

perimgDF$meanCorrectness<-perimgDF$meanCorrectness*100
perimgDF$accuracyStderr <- aggregate(x=ppantMeans$correct, by=list(ppantMeans$image),FUN=std.error)
accsem <- perimgDF$accuracyStderr
accsem <- accsem$x

perimgDF<- perimgDF[,c("image","meanCorrectness","meanConf")]
perimgDF$accstderr <- accsem
perimgDF$accstderr <- perimgDF$accstderr*100
  
perimgDF$meanConfAligned <- (perimgDF$meanConf/9)*100

perimgDF$confStdErr <- aggregate(x=ppantMeans$confidence, by=list(ppantMeans$image),FUN=std.error)

confsem <- perimgDF$confStdErr
confsem <- confsem$x
perimgDF<- perimgDF[,c("image","meanCorrectness","accstderr","meanConf","meanConfAligned")]
perimgDF$stderrConf <- confsem
perimgDF$stderrConfAligned <- (perimgDF$stderrConf/9)*100

perimgDF <- perimgDF %>%
  mutate(Realness = case_when(
    image<51 ~ "Real",
    image>50 ~ "Fake"
  ))

fig1 <- ggplot(perimgDF,aes(x=image,y=meanConfAligned))+
  geom_point(data=perimgDF,aes(fill=Realness))+
  geom_errorbar(data=perimgDF,aes(ymin=meanConfAligned-stderrConfAligned,ymax=meanConfAligned+stderrConfAligned),width=.3)+
  ylim(0,100)+
  geom_hline(yintercept=50, linetype="dashed", color = "#565656")+
  theme_minimal()+
  geom_point()

perimgDF$image <- factor(perimgDF$image, levels=order(-perimgDF$meanCorrectness))

fig2 <- ggplot() +
  geom_point(data=perimgDF,aes(x=image,y=meanConfAligned))+#,colour=Realness))+
  geom_vline(xintercept=0.5)+
  geom_errorbar(data=perimgDF,aes(x=image,y=meanConfAligned,ymin=meanConfAligned-stderrConfAligned,ymax=meanConfAligned+stderrConfAligned))+#,colour=Realness))+
  geom_point(data=perimgDF,aes(x=image,y=meanCorrectness,colour=Realness),size=2)+
  geom_hline(yintercept=50, linetype="dashed", color = "#565656")+
  geom_hline(yintercept=0)+
  geom_errorbar(data=perimgDF,aes(x=image,y=meanCorrectness,ymin=meanCorrectness-accstderr,ymax=meanCorrectness+accstderr,colour=Realness))+
  theme_minimal()+
  labs(x="Images (Rank-Ordered by Percentage Correct Responses)",
       y="Accuracy (%)",
       color='Image')+
  ylim(0,100)

fig2
ggplotly(fig2)

cor.test(perimgDF$meanConf, perimgDF$meanCorrectness, method='pearson')

# to get right hand axis of graphical figure:
az <- c(0:9)
bz <- c(0:9)
dz <- data.frame(az)
dz['bz']<-bz
aaz<- ggplot(dz, aes(x=az,y=bz))+geom_point()+
  theme_minimal()+
  labs(y="Confidence")+
  scale_y_continuous(n.breaks=9)
aaz
ggplotly(aaz)
