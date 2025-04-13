library(dplyr)
library(rstatix)
library(ggplot2)
library(readr)
library(effsize)
  

csv <- read_csv('280scores.csv')

A <- subset(csv,csv$Category == 'Control')
B <- subset(csv,csv$Category == 'Familiarization')
C <- subset(csv,csv$Category == 'One-Time Advice')
D <- subset(csv,csv$Category == 'Advice with Reminders')

mean(A$Score)*5
sd(A$Score)*5
mean(B$Score)*5
sd(B$Score)*5
mean(C$Score)*5
sd(C$Score)*5
mean(D$Score)*5
sd(D$Score)*5

mean(csv$Score)*5
sd(csv$Score)*5

aov(Score~Category, data=csv)

cohen.d(B$Score,A$Score)
cohen.d(C$Score,A$Score)
cohen.d(D$Score,A$Score)
cohen.d(C$Score,B$Score)
cohen.d(D$Score,B$Score)
cohen.d(D$Score,C$Score)

t.test(A$Score,mu=10)
t.test(B$Score,mu=10)
t.test(C$Score,mu=10)
t.test(D$Score,mu=10)

A %>% cohens_d(Score~1,mu=10)
B %>% cohens_d(Score~1,mu=10)
C %>% cohens_d(Score~1,mu=10)
D %>% cohens_d(Score~1,mu=10)
