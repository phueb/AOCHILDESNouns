library(lavaan)
library(semPlot)

#read data
data <- read.csv("scripts/quality/noun_quality_cs1_age_binned.csv")
qqnorm(data$selectivity, pch = 1, frame = FALSE)
qqline(data$selectivity, col = "steelblue", lwd = 2)

path <- " coverage ~ sem + syn
          selectivity ~ sem + syn
          prominence ~ sem + syn
          sem ~~ syn
          sem  ~ partition
          syn  ~ partition"

path <- " coverage ~ sem + syn
          selectivity ~ sem + syn
          prominence ~ sem + syn
          sem ~~ syn
          sem  ~ age
          syn  ~ age"

r <- sem(path, data=data)
summary(r, standardized = TRUE,fit.measures=TRUE,rsquare = TRUE)

semPaths(r,
what="std",
whatLabels="std",
intercepts=FALSE,
style="lisrel",
residuals=FALSE,
nCharNodes=0,
nCharEdges=0,
residScale=0,
layout="tree2",
curvePivot=FALSE)