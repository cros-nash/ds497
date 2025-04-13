library(dplyr)
library(plotly)
library(effsize)
library(ggplot2)
library(readr)


csv <- read_csv("eachadviceitemnumuses.csv")
head(csv)

newAdvDF <- data.frame("Condition"=character(0),"Correctness"=character(0),"Pct"=numeric(0),"NumUsed"=numeric(0),"NumReason"=numeric(0))
newAdvDF
for(i in unique(csv$Condition)){
  CORRECTnums <- 0
  CORRECTtotals <- 0
  INCORRECTnums <- 0
  INCORRECTtotals <- 0
  first_corr <- F
  first_incorr <- F
  for(row in 1:nrow(csv)){
    if(csv[row,2]==i){
      if(csv[row,3]=='Correct'){
        CORRECTnums <- CORRECTnums+ csv[row,5]
        if(!first_corr){
          CORRECTtotals <- CORRECTtotals+ csv[row,6]
          first_corr <- T
        }
      }else{
        INCORRECTnums <- INCORRECTnums+ csv[row,5]
        if(!first_incorr){
          INCORRECTtotals <- INCORRECTtotals+ csv[row,6]
          first_incorr <- T
        }
      }
    }
  }
  corrPct <- as.numeric(CORRECTnums/CORRECTtotals)*100
  incorrPct <- as.numeric(INCORRECTnums/INCORRECTtotals)*100
  tmpDF <- data.frame("Condition"=i,"Correctness"="Correct","Pct"=corrPct,"NumUsed"=CORRECTnums,"NumReason"=CORRECTtotals)
  tmpDF2 <- data.frame("Condition"=i,"Correctness"="Incorrect","Pct"=incorrPct,"NumUsed"=INCORRECTnums,"NumReason"=INCORRECTtotals)
  newAdvDF <- rbind(newAdvDF,tmpDF)
  newAdvDF <- rbind(newAdvDF,tmpDF2)
}

ggplot(newAdvDF, aes(x=Condition,y=Pct,fill=Correctness)) +
  geom_bar(stat="identity", position='dodge')+#, color="black") +
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 90, hjust=1, vjust=0.5)) +
  labs(y="Percentage of Participants' Reasoning that used an Advice Item",x='Condition')

csv%>%filter(Condition=='Advice with Reminders')

csv2 <- read_csv("eachadviceitemnumuses.csv")
csv2$Condition <- factor(csv2$Condition,
                        levels=c("Control","Familiarization","One-Time Advice","Advice with Reminders"))
print(unique(csv2$`Advice Item`))
csv2$`Advice Item`
csv2[80,4]
csv2$`Advice Item` <- factor(csv2$`Advice Item`,
        levels=c(csv2[80,4],"Impossible Location",'Abnormal Clothing Structure',
                'Strange Clothing Fabric', 'Distorted Text/Patterns',
                'Asymmetric Earrings','Colours Bleed From Background',
                "Asymmetric Facial Hair",'Warped Companion Faces',
                'Asymmetric Glasses'))
          
ggplot(csv2, aes(x=`Advice Item`,y=`Percentage`,fill=Correctness)) +
  geom_bar(stat="identity", position='stack')+#, color="black") +
        facet_wrap( ~ Condition)

ggplot(csv2, aes(x=`Advice Item`,y=`Percentage/8`,fill=Correctness)) +
  geom_bar(stat="identity", position='stack')+#, color="black") +
  theme(axis.text.x = element_text(angle = 90, hjust=1, vjust=0.5)) +
  labs(y="Percentage")

csv2['pctby4'] <- csv2$`Percentage/8`*2

csv3 <- csv2[,c('Condition','Correctness','Advice Item','Number of Reasonings','NumResponses')]

csv3 %>% filter(`Advice Item` == 'Warped Companion Faces')

newAdvDF <- data.frame("AdviceItem"=character(0),"Correctness"=character(0),"Pct"=numeric(0))

for(i in unique(csv3$`Advice Item`)){
  CORRECTnums <- 0
  CORRECTtotals <- 0
  INCORRECTnums <- 0
  INCORRECTtotals <- 0
  for(row in 1:nrow(csv3)){
    if(csv3[row,3]==i){
      if(csv3[row,2]=='Correct'){
        CORRECTnums <- CORRECTnums+ csv3[row,4]
        CORRECTtotals <- CORRECTtotals+ csv3[row,5]
      }else{
        INCORRECTnums <- INCORRECTnums+ csv3[row,4]
        INCORRECTtotals <- INCORRECTtotals+ csv3[row,5]
      }
    }
  }
  corrPct <- as.numeric(CORRECTnums/CORRECTtotals)*100
  incorrPct <- as.numeric(INCORRECTnums/INCORRECTtotals)*100
  tmpDF <- data.frame("AdviceItem"=i,"Correctness"="Correct","Pct"=corrPct)
  tmpDF2 <- data.frame("AdviceItem"=i,"Correctness"="Incorrect","Pct"=incorrPct)
  newAdvDF <- rbind(newAdvDF,tmpDF)
  newAdvDF <- rbind(newAdvDF,tmpDF2)
}

ggplot(newAdvDF, aes(x=reorder(AdviceItem,-Pct),y=Pct,fill=Correctness)) +
  geom_bar(stat="identity", position='dodge')+#, color="black") +
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 90, hjust=1, vjust=0.5)) +
  labs(y='Percentage Of Responses That Contained Each Advice Item',
       x='Advice Item')
