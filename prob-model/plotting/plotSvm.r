library("foreign")
X11.options(type="Xlib") 

# SVM hit rate
filename <- "svm_hit_rate";
svm_hit_rate <- read.octave(paste("serial/", filename, ".mat", sep=""))
svm_hit_rate <- svm_hit_rate[[1]]
pdf(file=paste(filename, ".pdf", sep=""), width=7, height=4)
last <- length(svm_hit_rate[1, ])
last
dim(svm_hit_rate)
plot(2:last, svm_hit_rate[1, 2:last], type="l", col="black", ylim=c(0, 1), xlab="Iteration", ylab="Hit rate", lty=1, lwd=2)
lines(2:last, svm_hit_rate[2, 2:last], type="l", col="blue", pch=22, lty=2, lwd=2)
lines(2:last, svm_hit_rate[3, 2:last], type="l", col="red", pch=22, lty=3, lwd=2)
lines(2:last, svm_hit_rate[4, 2:last], type="l", col="green", pch=22, lty=4, lwd=2)
title(main="SVM Quality", col.main="black", font.main=4)
legend(x="bottomright", legend=c("Accuracy", "Precision", "Recall", "F-measure"), col=c("black", "blue", "red", "green"), lty=1:3, lwd=2)
dev.off()

# SVM training times
filename <- "svm_training_serial";
svm_training_serial <- read.octave(paste("serial/", filename, ".mat", sep=""))
svm_training_serial <- svm_training_serial[[1]]
last <- length(svm_training_serial)
pdf(file=paste(filename, ".pdf", sep=""), width=7, height=4)
plot(2:last, svm_training_serial[2:last], type="l", col="black", xlab="Iteration", ylab="Training time [s]", lty=1, lwd=2)
title(main="SVM training times", col.main="black", font.main=4)
dev.off()

# SVM prediction times
filename <- "svm_prediction_serial";
svm_prediction_serial <- read.octave(paste("serial/", filename, ".mat", sep=""))
svm_prediction_serial <- svm_prediction_serial[[1]]
last <- length(svm_prediction_serial)
pdf(file=paste(filename, ".pdf", sep=""), width=7, height=4)
plot(2:last, svm_prediction_serial[2:last], type="l", col="black", xlab="Iteration", ylab="Prediction time [s]", lty=1, lwd=2)
title(main="SVM prediction times", col.main="black", font.main=4)
dev.off()

