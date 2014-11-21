require(ggplot2)
require(reshape)

dat <- read.csv('Model1_preds.csv')
dd <- data.frame(dat[, -1])
names(dd) <- names(dat)[2:ncol(dat)]
df <- melt(dd, id.vars=c())

plt <- ggplot(df, aes(x=value, fill=variable))
plt + geom_density(alpha=0.5) + xlab('Learning Rate') + ylab('Probability Density')

ggsave('Model1_alpha_predicted_density.pdf', width=6, height=4, units='in')
