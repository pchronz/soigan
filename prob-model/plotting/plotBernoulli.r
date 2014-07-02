library("foreign")
X11.options(type="Xlib") 

# Bernoulli hit rate
filename <- "bernoulli_hit_rate";
bernoulli_hit_rate <- read.octave(paste("serial/", filename, ".mat", sep=""))
bernoulli_hit_rate <- bernoulli_hit_rate[[1]]
pdf(file=paste(filename, ".pdf", sep=""), width=7, height=4)
last <- length(bernoulli_hit_rate[1, ])
plot(2:last, bernoulli_hit_rate[1, 2:last], type="l", col="black", ylim=c(0, 1), xlab="Iteration", ylab="Hit rate", lty=1, lwd=2)
lines(2:last, bernoulli_hit_rate[2, 2:last], type="l", col="blue", pch=22, lty=2, lwd=2)
lines(2:last, bernoulli_hit_rate[3, 2:last], type="l", col="red", pch=22, lty=3, lwd=2)
lines(2:last, bernoulli_hit_rate[4, 2:last], type="l", col="green", pch=22, lty=4, lwd=2)
title(main="Bernoulli Quality", col.main="black", font.main=4)
legend(x="bottomright", legend=c("Accuracy", "Precision", "Recall", "F-measure"), col=c("black", "blue", "red", "green"), lty=1:3, lwd=2)
dev.off()

# Bernoulli training times
filename <- "bernoulli_training_serial";
bernoulli_training_serial <- read.octave(paste("serial/", filename, ".mat", sep=""))
bernoulli_training_serial <- bernoulli_training_serial[[1]]
last <- length(bernoulli_training_serial)
pdf(file=paste(filename, ".pdf", sep=""), width=7, height=4)
plot(2:last, bernoulli_training_serial[2:last], type="l", col="black", xlab="Iteration", ylab="Training time [s]", lty=1, lwd=2)
title(main="Bernoulli training times", col.main="black", font.main=4)
dev.off()

# Bernoulli prediction times
filename <- "bernoulli_prediction_serial";
bernoulli_prediction_serial <- read.octave(paste("serial/", filename, ".mat", sep=""))
bernoulli_prediction_serial <- bernoulli_prediction_serial[[1]]
last <- length(bernoulli_prediction_serial)
pdf(file=paste(filename, ".pdf", sep=""), width=7, height=4)
plot(2:last, bernoulli_prediction_serial[2:last], type="l", col="black", xlab="Iteration", ylab="Prediction time [s]", lty=1, lwd=2)
title(main="Bernoulli prediction times", col.main="black", font.main=4)
dev.off()

