library("foreign")
X11.options(type="Xlib") 

# SVM, prob model, bernoulli, and clustering accuracy comparison
pdf(file=paste("accuracy-comparison", ".pdf", sep=""), width=7, height=4)
# SVM
filename <- "svm_hit_rate";
svm_hit_rate <- read.octave(paste("serial/", filename, ".mat", sep=""))
svm_hit_rate <- svm_hit_rate[[1]]
last <- length(svm_hit_rate[1, ])
plot(2:last, svm_hit_rate[1, 2:last], type="l", col="black", ylim=c(0, 1), xlab="Iteration", ylab="Accuracy", lty=1, lwd=2)
# Bernoulli
filename <- "bernoulli_hit_rate";
bernoulli_hit_rate <- read.octave(paste("serial/", filename, ".mat", sep=""))
bernoulli_hit_rate <- bernoulli_hit_rate[[1]]
last <- length(bernoulli_hit_rate[1, ])
lines(2:last, bernoulli_hit_rate[1, 2:last], type="l", col="red", ylim=c(0, 1), lty=2, lwd=2)
# Baseline
filename <- "baseline_hit_rate_serial";
baseline_hit_rate_serial <- read.octave(paste("serial/", filename, ".mat", sep=""))
baseline_hit_rate_serial <- baseline_hit_rate_serial[[1]]
last <- length(baseline_hit_rate_serial[max_K, , 1])
best_K <- which.max(baseline_hit_rate_serial[, last, 1])
lines(2:last, baseline_hit_rate_serial[best_K, 2:last, 1], type="l", col="blue", ylim=c(0, 1), lty=3, lwd=2)
# Prob model
filename <- "prob_model_hit_rate_serial";
prob_model_hit_rate_serial <- read.octave(paste("serial/", filename, ".mat", sep=""))
prob_model_hit_rate_serial <- prob_model_hit_rate_serial[[1]]
last <- length(prob_model_hit_rate_serial[max_K, , 1])
best_K_GMM <- which.max(prob_model_hit_rate_serial[, last, 1])
lines(2:last, prob_model_hit_rate_serial[best_K_GMM, 2:last, 1], type="l", col="green", ylim=c(0, 1), lty=4, lwd=2)
# Plot out
title(main="Accuracy Comparison", col.main="black", font.main=4)
legend(x="bottomright", legend=c("SVM", "Bernoulli", paste("Clustering, K=", best_K), paste("GMM, K=", best_K_GMM)), col=c("black", "red", "blue", "green"), lty=c(1:4), lwd=2)
dev.off()

# SVM, prob model, bernoulli, and clustering precision comparison
pdf(file=paste("precision-comparison", ".pdf", sep=""), width=7, height=4)
# SVM
filename <- "svm_hit_rate";
svm_hit_rate <- read.octave(paste("serial/", filename, ".mat", sep=""))
svm_hit_rate <- svm_hit_rate[[1]]
last <- length(svm_hit_rate[1, ])
plot(2:last, svm_hit_rate[2, 2:last], type="l", col="black", ylim=c(0, 1), xlab="Iteration", ylab="Precision", lty=1, lwd=2)
# Bernoulli
filename <- "bernoulli_hit_rate";
bernoulli_hit_rate <- read.octave(paste("serial/", filename, ".mat", sep=""))
bernoulli_hit_rate <- bernoulli_hit_rate[[1]]
last <- length(bernoulli_hit_rate[1, ])
lines(2:last, bernoulli_hit_rate[2, 2:last], type="l", col="red", ylim=c(0, 1), lty=2, lwd=2)
# Baseline
filename <- "baseline_hit_rate_serial";
baseline_hit_rate_serial <- read.octave(paste("serial/", filename, ".mat", sep=""))
baseline_hit_rate_serial <- baseline_hit_rate_serial[[1]]
last <- length(baseline_hit_rate_serial[best_K, , 3])
lines(2:last, baseline_hit_rate_serial[best_K, 2:last, 2], type="l", col="blue", ylim=c(0, 1), lty=3, lwd=2)
# Prob model
filename <- "prob_model_hit_rate_serial";
prob_model_hit_rate_serial <- read.octave(paste("serial/", filename, ".mat", sep=""))
prob_model_hit_rate_serial <- prob_model_hit_rate_serial[[1]]
last <- length(prob_model_hit_rate_serial[best_K_GMM, , 3])
lines(2:last, prob_model_hit_rate_serial[best_K, 2:last, 2], type="l", col="green", ylim=c(0, 1), lty=4, lwd=2)
# Plot out
title(main="Precision Comparison", col.main="black", font.main=4)
legend(x="bottomright", legend=c("SVM", "Bernoulli", paste("Clustering, K=", best_K), paste("GMM, K=", best_K_GMM)), col=c("black", "red", "blue", "green"), lty=c(1:4), lwd=2)
dev.off()

# SVM, prob model, bernoulli, and clustering recall comparison
pdf(file=paste("recall-comparison", ".pdf", sep=""), width=7, height=4)
# SVM
filename <- "svm_hit_rate";
svm_hit_rate <- read.octave(paste("serial/", filename, ".mat", sep=""))
svm_hit_rate <- svm_hit_rate[[1]]
last <- length(svm_hit_rate[1, ])
plot(2:last, svm_hit_rate[3, 2:last], type="l", col="black", ylim=c(0, 1), xlab="Iteration", ylab="Recall", lty=1, lwd=2)
# Bernoulli
filename <- "bernoulli_hit_rate";
bernoulli_hit_rate <- read.octave(paste("serial/", filename, ".mat", sep=""))
bernoulli_hit_rate <- bernoulli_hit_rate[[1]]
last <- length(bernoulli_hit_rate[1, ])
lines(2:last, bernoulli_hit_rate[3, 2:last], type="l", col="red", ylim=c(0, 1), lty=2, lwd=2)
# Baseline
filename <- "baseline_hit_rate_serial";
baseline_hit_rate_serial <- read.octave(paste("serial/", filename, ".mat", sep=""))
baseline_hit_rate_serial <- baseline_hit_rate_serial[[1]]
last <- length(baseline_hit_rate_serial[best_K, , 3])
lines(2:last, baseline_hit_rate_serial[best_K, 2:last, 3], type="l", col="blue", ylim=c(0, 1), lty=3, lwd=2)
# Prob model
filename <- "prob_model_hit_rate_serial";
prob_model_hit_rate_serial <- read.octave(paste("serial/", filename, ".mat", sep=""))
prob_model_hit_rate_serial <- prob_model_hit_rate_serial[[1]]
last <- length(prob_model_hit_rate_serial[best_K_GMM, , 3])
lines(2:last, prob_model_hit_rate_serial[best_K_GMM, 2:last, 3], type="l", col="green", ylim=c(0, 1), lty=4, lwd=2)
# Plot out
title(main="Recall Comparison", col.main="black", font.main=4)
legend(x="bottomright", legend=c("SVM", "Bernoulli", paste("Clustering, K=", best_K), paste("GMM, K=", best_K_GMM)), col=c("black", "red", "blue", "green"), lty=c(1:4), lwd=2)
dev.off()

# SVM, prob model, bernoulli, and clustering F-measure comparison
pdf(file=paste("f-measure-comparison", ".pdf", sep=""), width=7, height=4)
# SVM
filename <- "svm_hit_rate";
svm_hit_rate <- read.octave(paste("serial/", filename, ".mat", sep=""))
svm_hit_rate <- svm_hit_rate[[1]]
last <- length(svm_hit_rate[1, ])
plot(2:last, svm_hit_rate[4, 2:last], type="l", col="black", ylim=c(0, 1), xlab="Iteration", ylab="F-measure", lty=1, lwd=2)
# Bernoulli
filename <- "bernoulli_hit_rate";
bernoulli_hit_rate <- read.octave(paste("serial/", filename, ".mat", sep=""))
bernoulli_hit_rate <- bernoulli_hit_rate[[1]]
last <- length(bernoulli_hit_rate[1, ])
lines(2:last, bernoulli_hit_rate[4, 2:last], type="l", col="red", ylim=c(0, 1), lty=2, lwd=2)
# Baseline
filename <- "baseline_hit_rate_serial";
baseline_hit_rate_serial <- read.octave(paste("serial/", filename, ".mat", sep=""))
baseline_hit_rate_serial <- baseline_hit_rate_serial[[1]]
last <- length(baseline_hit_rate_serial[best_K, , 4])
lines(2:last, baseline_hit_rate_serial[best_K, 2:last, 4], type="l", col="blue", ylim=c(0, 1), lty=3, lwd=2)
# Prob model
filename <- "prob_model_hit_rate_serial";
prob_model_hit_rate_serial <- read.octave(paste("serial/", filename, ".mat", sep=""))
prob_model_hit_rate_serial <- prob_model_hit_rate_serial[[1]]
last <- length(prob_model_hit_rate_serial[best_K_GMM, , 4])
lines(2:last, prob_model_hit_rate_serial[best_K_GMM, 2:last, 4], type="l", col="green", ylim=c(0, 1), lty=4, lwd=2)
# Plot out
title(main="F-measure Comparison", col.main="black", font.main=4)
legend(x="bottomright", legend=c("SVM", "Bernoulli", paste("Clustering, K=", best_K), paste("GMM, K=", best_K)), col=c("black", "red", "blue", "green"), lty=c(1:4), lwd=2)
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


