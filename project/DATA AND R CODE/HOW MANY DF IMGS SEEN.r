library(dplyr)
library(rstatix)
library(ggplot2)
library(readr)
library(effsize)
  

csv <- read_csv('280scores.csv')
csv <- read_csv('eachresponse.csv')

justThese <- csv[,c("ppantID","choice","correct","intervention")]
guessReal <- justThese[justThese$choice=='real',]
guessFake <- justThese[justThese$choice=='ai',]

actualReal1 <- guessReal %>% filter(correct=="correct")
actualReal2 <- guessFake %>% filter(correct=="incorrect")
actualFake1 <- guessReal %>% filter(correct=="incorrect")
actualFake2 <- guessFake %>% filter(correct=="correct")

actualReal <- rbind(actualReal1,actualReal2)
actualFake <- rbind(actualFake1,actualFake2)

AF <- subset(actualFake,actualFake$intervention == 'A')
BF <- subset(actualFake,actualFake$intervention == 'B')
CF <- subset(actualFake,actualFake$intervention == 'C')
DF <- subset(actualFake,actualFake$intervention == 'D')

TFULL<-table(actualFake$ppantID)
mean(TFULL)
sd(TFULL)
TA<-table(AF$ppantID)
mean(TA)
sd(TA)
TB<-table(BF$ppantID)
mean(TB)
sd(TB)
TC<-table(CF$ppantID)
mean(TC)
sd(TC)
TD<-table(DF$ppantID)
mean(TD)
sd(TD)
