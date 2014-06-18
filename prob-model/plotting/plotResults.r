library("foreign")
X11.options(type="Xlib") 

# baseline hit rates for Ks separate
filename <- "baseline_hit_rate_serial";
baseline_hit_rate_serial <- read.octave(paste("serial/", filename, ".mat", sep=""))
baseline_hit_rate_serial <- baseline_hit_rate_serial[[1]]
max_K = length(baseline_hit_rate_serial[, 1, 1])
for(K in 2:max_K) {
  pdf(file=paste(filename, "-K", K, ".pdf", sep=""), width=7, height=4)
  last <- length(baseline_hit_rate_serial[K, , 1])
  plot(2:last, baseline_hit_rate_serial[K, 2:last, 1], type="l", col="black", ylim=c(0, 1), xlab="Iteration", ylab="Hit rate", lty=1, lwd=2)
  lines(2:last, baseline_hit_rate_serial[K, 2:last, 2], type="l", col="blue", pch=22, lty=2, lwd=2)
  lines(2:last, baseline_hit_rate_serial[K, 2:last, 3], type="l", col="red", pch=22, lty=3, lwd=2)
  lines(2:last, baseline_hit_rate_serial[K, 2:last, 4], type="l", col="green", pch=22, lty=4, lwd=2)
  title(main="Clustering Quality", col.main="black", font.main=4)
  legend(x="bottomright", legend=c("Accuracy", "Precision", "Recall", "F-measure"), col=c("black", "blue", "red", "green"), lty=1:3, lwd=2)
  dev.off()
}

## plot total rates for various K for comparison
#pdf(file=paste(filename, "-K-comparison", ".pdf", sep=""), width=7, height=4)
#last <- length(baseline_hit_rate_serial[2, , 1])
#plot(2:last, baseline_hit_rate_serial[2, 2:last, 1], type="l", col="black", ylim=c(0, 1), xlab="Iteration", ylab="Hit rate", lty=1, lwd=2)
#for(K in 3:max_K) {
#  lines(2:last, baseline_hit_rate_serial[K, 2:last, 1], type="l", col="black", lty=K-1, lwd=2)
#}
#title(main="Clustering Hit Rate", col.main="black", font.main=4)
#legend(x="bottomright", legend=c(2:max_K), lty=c(1:(max_K-1)), lwd=2)
#dev.off()

# learning run times
filename <- "baseline_training_serial";
baseline_training_serial <- read.octave(paste("serial/", filename, ".mat", sep=""))
baseline_training_serial <- baseline_training_serial[[1]]
max_K = length(baseline_training_serial[, 1])
last <- length(baseline_training_serial[2, ])
maxY <- max(baseline_training_serial)
for(K in 2:max_K) {
  pdf(file=paste(filename, "-K_", K, ".pdf", sep=""), width=7, height=4)
  plot(2:last, baseline_training_serial[K, 2:last], ylim=c(0, maxY), type="l", col="black", xlab="Iteration", ylab="Training time [s]", lty=1, lwd=2)
  title(main=paste("Clustering training times, K = ", K), col.main="black", font.main=4)
  dev.off()
}

# clustering prediction times
filename <- "baseline_prediction_serial";
baseline_prediction_serial <- read.octave(paste("serial/", filename, ".mat", sep=""))
baseline_prediction_serial <- baseline_prediction_serial[[1]]
max_K = length(baseline_prediction_serial[, 1])
last <- length(baseline_prediction_serial[2, ])
maxY <- max(baseline_prediction_serial)
for(K in 2:max_K) {
  pdf(file=paste(filename, "-K_", K, ".pdf", sep=""), width=7, height=4)
  plot(2:last, baseline_prediction_serial[K, 2:last], ylim=c(0, maxY), type="l", col="black", xlab="Iteration", ylab="Prediction time [s]", lty=1, lwd=2)
  title(main=paste("Clustering prediction times, K = ", K), col.main="black", font.main=4)
  dev.off()
}

# SVM hit rate
filename <- "svm_hit_rate";
svm_hit_rate <- read.octave(paste("serial/", filename, ".mat", sep=""))
svm_hit_rate <- svm_hit_rate[[1]]
pdf(file=paste(filename, ".pdf", sep=""), width=7, height=4)
last <- length(svm_hit_rate[1, ])
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

# SVM and clustering accuracy comparison
filename <- "svm_hit_rate";
svm_hit_rate <- read.octave(paste("serial/", filename, ".mat", sep=""))
svm_hit_rate <- svm_hit_rate[[1]]
filename <- "bernoulli_hit_rate";
bernoulli_hit_rate <- read.octave(paste("serial/", filename, ".mat", sep=""))
bernoulli_hit_rate <- bernoulli_hit_rate[[1]]
filename <- "baseline_hit_rate_serial";
baseline_hit_rate_serial <- read.octave(paste("serial/", filename, ".mat", sep=""))
baseline_hit_rate_serial <- baseline_hit_rate_serial[[1]]
pdf(file=paste("accuracy-comparison", ".pdf", sep=""), width=7, height=4)
last <- length(svm_hit_rate[1, ])
plot(2:last, svm_hit_rate[1, 2:last], type="l", col="black", ylim=c(0, 1), xlab="Iteration", ylab="Accuracy", lty=1, lwd=2)
last <- length(bernoulli_hit_rate[1, ])
lines(2:last, bernoulli_hit_rate[1, 2:last], type="l", col="red", ylim=c(0, 1), lty=2, lwd=2)
last <- length(baseline_hit_rate_serial[max_K, , 1])
best_K <- which.max(baseline_hit_rate_serial[, last, 1])
lines(2:last, baseline_hit_rate_serial[best_K, 2:last, 1], type="l", col="blue", ylim=c(0, 1), lty=3, lwd=2)
title(main="Accuracy Comparison", col.main="black", font.main=4)
legend(x="bottomright", legend=c("SVM", "Bernoulli", paste("Clustering, K=", best_K)), col=c("black", "red", "blue"), lty=c(1:3), lwd=2)
dev.off()

# SVM and clustering precision comparison
filename <- "svm_hit_rate";
svm_hit_rate <- read.octave(paste("serial/", filename, ".mat", sep=""))
svm_hit_rate <- svm_hit_rate[[1]]
filename <- "bernoulli_hit_rate";
bernoulli_hit_rate <- read.octave(paste("serial/", filename, ".mat", sep=""))
bernoulli_hit_rate <- bernoulli_hit_rate[[1]]
filename <- "baseline_hit_rate_serial";
baseline_hit_rate_serial <- read.octave(paste("serial/", filename, ".mat", sep=""))
baseline_hit_rate_serial <- baseline_hit_rate_serial[[1]]
pdf(file=paste("precision-comparison", ".pdf", sep=""), width=7, height=4)
last <- length(svm_hit_rate[1, ])
plot(2:last, svm_hit_rate[2, 2:last], type="l", col="black", ylim=c(0, 1), xlab="Iteration", ylab="Precision", lty=1, lwd=2)
last <- length(bernoulli_hit_rate[1, ])
lines(2:last, bernoulli_hit_rate[2, 2:last], type="l", col="red", ylim=c(0, 1), lty=2, lwd=2)
last <- length(baseline_hit_rate_serial[best_K, , 3])
lines(2:last, baseline_hit_rate_serial[best_K, 2:last, 2], type="l", col="blue", ylim=c(0, 1), lty=3, lwd=2)
title(main="Precision Comparison", col.main="black", font.main=4)
legend(x="bottomright", legend=c("SVM", "Bernoulli", paste("Clustering, K=", best_K)), col=c("black", "red", "blue"), lty=c(1:3), lwd=2)
dev.off()

# SVM and clustering recall comparison
filename <- "svm_hit_rate";
svm_hit_rate <- read.octave(paste("serial/", filename, ".mat", sep=""))
svm_hit_rate <- svm_hit_rate[[1]]
filename <- "bernoulli_hit_rate";
bernoulli_hit_rate <- read.octave(paste("serial/", filename, ".mat", sep=""))
bernoulli_hit_rate <- bernoulli_hit_rate[[1]]
filename <- "baseline_hit_rate_serial";
baseline_hit_rate_serial <- read.octave(paste("serial/", filename, ".mat", sep=""))
baseline_hit_rate_serial <- baseline_hit_rate_serial[[1]]
pdf(file=paste("recall-comparison", ".pdf", sep=""), width=7, height=4)
last <- length(svm_hit_rate[1, ])
plot(2:last, svm_hit_rate[3, 2:last], type="l", col="black", ylim=c(0, 1), xlab="Iteration", ylab="Recall", lty=1, lwd=2)
last <- length(bernoulli_hit_rate[1, ])
lines(2:last, bernoulli_hit_rate[3, 2:last], type="l", col="red", ylim=c(0, 1), lty=2, lwd=2)
last <- length(baseline_hit_rate_serial[best_K, , 3])
lines(2:last, baseline_hit_rate_serial[best_K, 2:last, 3], type="l", col="blue", ylim=c(0, 1), lty=3, lwd=2)
title(main="Recall Comparison", col.main="black", font.main=4)
legend(x="bottomright", legend=c("SVM", "Bernoulli", paste("Clustering, K=", best_K)), col=c("black", "red", "blue"), lty=c(1:3), lwd=2)
dev.off()

# SVM and clustering F-measure comparison
filename <- "svm_hit_rate";
svm_hit_rate <- read.octave(paste("serial/", filename, ".mat", sep=""))
svm_hit_rate <- svm_hit_rate[[1]]
filename <- "bernoulli_hit_rate";
bernoulli_hit_rate <- read.octave(paste("serial/", filename, ".mat", sep=""))
bernoulli_hit_rate <- bernoulli_hit_rate[[1]]
filename <- "baseline_hit_rate_serial";
baseline_hit_rate_serial <- read.octave(paste("serial/", filename, ".mat", sep=""))
baseline_hit_rate_serial <- baseline_hit_rate_serial[[1]]
pdf(file=paste("f-measure-comparison", ".pdf", sep=""), width=7, height=4)
last <- length(svm_hit_rate[1, ])
plot(2:last, svm_hit_rate[4, 2:last], type="l", col="black", ylim=c(0, 1), xlab="Iteration", ylab="F-measure", lty=1, lwd=2)
last <- length(bernoulli_hit_rate[1, ])
lines(2:last, bernoulli_hit_rate[4, 2:last], type="l", col="red", ylim=c(0, 1), lty=2, lwd=2)
last <- length(baseline_hit_rate_serial[best_K, , 4])
lines(2:last, baseline_hit_rate_serial[best_K, 2:last, 4], type="l", col="blue", ylim=c(0, 1), lty=3, lwd=2)
title(main="F-measure Comparison", col.main="black", font.main=4)
legend(x="bottomright", legend=c("SVM", "Bernoulli", paste("Clustering, K=", best_K)), col=c("black", "red", "blue"), lty=c(1:3), lwd=2)
dev.off()

# baseline & SVM accuracy
#filename <- "baseline_accuracy"
#baseline_accuracy <- read.octave(paste("parallel/", filename, ".mat", sep=""))
#baseline_accuracy <- baseline_accuracy[[1]]
#filename <- "svm_accuracy"
#svm_accuracy <- read.octave(paste("parallel/", filename, ".mat", sep=""))
#svm_accuracy <- svm_accuracy[[1]]
#last <- length(baseline_accuracy[1, , 1])
#max_K <- length(baseline_accuracy[1, 1, ])
#pdf(file=paste("accuracies", ".pdf", sep=""), width=7, height=4)
#accuracies <- cbind(baseline_accuracy[1, 2:last, 2:max_K], svm_accuracy[1, 2:last, 2])
#names <- vector()
#for(K in 2:max_K) {
#  names[K-1] <- paste("K=", K)
#}
#names[max_K] <- "SVM"
#boxplot(accuracies, xlab="Method", ylab="Accuracy", names=names)
#title(main=paste("Accuracies (parallel)"), col.main="black", font.main=4)
#dev.off()

# baseline & SVM training
#filename <- "baseline_learning"
#baseline_learning <- read.octave(paste("parallel/", filename, ".mat", sep=""))
#baseline_learning <- baseline_learning[[1]]
#filename <- "svm_learning"
#svm_learning <- read.octave(paste("parallel/", filename, ".mat", sep=""))
#svm_learning <- svm_learning[[1]]
#last <- length(baseline_learning[1, , 1])
#max_K <- length(baseline_learning[1, 1, ])
#pdf(file=paste("learning-durations", ".pdf", sep=""), width=7, height=4)
#accuracies <- cbind(baseline_learning[1, 2:last, 2:max_K], svm_learning[1, 2:last, 2])
#names <- vector()
#for(K in 2:max_K) {
#  names[K-1] <- paste("K=", K)
#}
#names[max_K] <- "SVM"
#boxplot(accuracies, xlab="Method", ylab="Learning durations", names=names)
#title(main=paste("Learning Durations (parallel)"), col.main="black", font.main=4)
#dev.off()

# baseline & SVM prediction
#filename <- "baseline_prediction"
#baseline_prediction <- read.octave(paste("parallel/", filename, ".mat", sep=""))
#baseline_prediction <- baseline_prediction[[1]]
#filename <- "svm_prediction"
#svm_prediction <- read.octave(paste("parallel/", filename, ".mat", sep=""))
#svm_prediction <- svm_prediction[[1]]
#last <- length(baseline_prediction[1, , 1])
#max_K <- length(baseline_prediction[1, 1, ])
#pdf(file=paste("prediction-durations", ".pdf", sep=""), width=7, height=4)
#accuracies <- cbind(baseline_prediction[1, 2:last, 2:max_K], svm_prediction[1, 2:last, 2])
#names <- vector()
#for(K in 2:max_K) {
#  names[K-1] <- paste("K=", K)
#}
#names[max_K] <- "SVM"
#boxplot(accuracies, xlab="Method", ylab="Prediction durations", names=names)
#title(main=paste("Prediction Durations (parallel)"), col.main="black", font.main=4)
#dev.off()


