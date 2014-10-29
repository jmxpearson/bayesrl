require(ggplot2)
require(reshape)

dat <- read.csv('Model1_preds.csv')
df <- melt(dat[,2:3], id.vars=c())

plt <- ggplot(df, aes(x=value, fill=variable))
plt + geom_density(alpha=0.5)
