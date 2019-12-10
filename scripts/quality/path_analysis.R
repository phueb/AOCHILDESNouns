library(lavaan)
library(semPlot)

#read data
data <- read.csv("scripts/quality/noun_quality_cs2_np32.csv")
qqnorm(data$coverage, pch = 1, frame = FALSE)
qqline(data$coverage, col = "steelblue", lwd = 2)

path <- " coverage ~ sem + syn
                  sem  ~ partition
                  syn  ~ partition"


r <- sem(path, data=data)
summary(r, standardized = TRUE,fit.measures=TRUE,rsquare = TRUE)

pathdiagram<-semPaths(r,
whatLabels="std",
intercepts=FALSE,
style="lisrel",
nCharNodes=0,
nCharEdges=0,
curveAdjacent = TRUE,
title=TRUE,
layout="tree2",
curvePivot=TRUE)