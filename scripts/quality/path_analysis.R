library(lavaan)

#read data
data <- read.csv("scripts/quality/noun_quality.csv")

path <- " selectivity ~ partition + sem.comp + syn.comp
                  sem.comp  ~ partition
                  syn.comp  ~ partition"

r <- sem(path, data=data)
summary(r, standardized = TRUE,fit.measures=TRUE)

