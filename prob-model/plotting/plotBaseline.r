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





