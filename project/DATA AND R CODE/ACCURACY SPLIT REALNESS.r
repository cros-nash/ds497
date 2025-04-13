library("tidyverse")
library("readr")
library("dplyr")


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

actualReal$point[actualReal$correct =='correct'] <- as.integer(1)
actualReal$point[actualReal$correct =='incorrect'] <- as.integer(0)
actualFake$point[actualFake$correct =='correct'] <- as.integer(1)
actualFake$point[actualFake$correct =='incorrect'] <- as.integer(0)

actualReal$counting <- 1
aggregReal <- aggregate(point~ppantID, data=actualReal, sum)
aggregReal$totals <- aggregate(counting~ppantID, data=actualReal, sum)$counting
aggregReal <- aggregReal %>% select(ppantID,point,totals) %>%
  mutate(means = 100*(point/totals))

actualFake$counting <- 1
aggregFake <- aggregate(point~ppantID, data=actualFake, sum)
aggregFake$totals <- aggregate(counting~ppantID, data=actualFake, sum)$counting
aggregFake <- aggregFake %>% select(ppantID,point,totals) %>%
  mutate(means = 100*(point/totals))

aggregReal$intervention <- aggregate(intervention~ppantID,data=actualReal,min)$intervention
aggregFake$intervention <- aggregate(intervention~ppantID,data=actualFake,min)$intervention
aggregReal$realness <- 1
aggregFake$realness <- 0
aggregBoth <- rbind(aggregReal,aggregFake)

dblanov <- aov(means ~ realness * intervention, data=aggregBoth)
summary(dblanov)

aggregate(means~intervention,data=aggregReal,mean)
aggregate(means~intervention,data=aggregReal,sd)
aggregate(means~intervention,data=aggregFake,mean)
aggregate(means~intervention,data=aggregFake,sd)

mean(aggregReal$means)
sd(aggregReal$means)
mean(aggregFake$means)
sd(aggregFake$means)
