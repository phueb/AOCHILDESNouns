library(lavaan)
library(semPlot)

#read data
data <- read.csv("scripts/quality/noun_quality_cs1_np32.csv")
qqnorm(data$coverage, pch = 1, frame = FALSE)
qqline(data$coverage, col = "steelblue", lwd = 2)

path <- " coverage ~ sem + syn
          selectivity ~ sem + syn
          prominence ~ sem + syn
          sem ~~ syn
          sem  ~ partition
          syn  ~ partition"


r <- sem(path, data=data)
summary(r, standardized = TRUE,fit.measures=TRUE,rsquare = TRUE)

pathdiagram<-semPaths(r,
    what="std",
whatLabels="no",
intercepts=FALSE,
style="lisrel",
    residulas=FALSE,
nCharNodes=0,
nCharEdges=0,
residScalse=0,
layout="tree2",
curvePivot=FALSE)