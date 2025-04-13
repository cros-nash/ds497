library(dplyr)
library(plyr)
library(effsize)
library(plotly)
library(ggplot2)
library(readr)
library(reshape2)


csv <- read_csv('adviceappliedornot.csv')

csv$Condition <- factor(csv$Condition,levels=c("Control","Familiarization","One-Time Advice","Advice With Reminders"))

gg1 <- ggplot(csv, aes(x=Condition,y=`%ge`,fill=Correctness))+
  geom_bar(stat='identity',position='dodge')+
  lims(y=c(0,100))+
  theme_minimal()+
  labs(y='Percentage Of Responses That Contained Advice Items',fill='Correctness')
gg1
