library(dplyr)
library(ggplot2)
library(readr)


csv <- read_csv('280scores.csv')

mean(csv$num_said_DF)
sd(csv$num_said_DF)

anova1 <- aov(num_said_DF~Category, data=csv)
summary(anova1)

A <- subset(csv,csv$Category == 'Control')
B <- subset(csv,csv$Category == 'Familiarization')
C <- subset(csv,csv$Category == 'One-Time Advice')
D <- subset(csv,csv$Category == 'Advice with Reminders')

mean(A$num_said_DF)
sd(A$num_said_DF)
mean(B$num_said_DF)
sd(B$num_said_DF)
mean(C$num_said_DF)
sd(C$num_said_DF)
mean(D$num_said_DF)
sd(D$num_said_DF)

t.test(A$num_said_DF,mu=10.03)
t.test(B$num_said_DF,mu=10.18)
t.test(C$num_said_DF,mu=9.82)
t.test(D$num_said_DF,mu=9.98)

anova2 <- aov(num_said_DF~Category, data=csv)
summary(anova2)
