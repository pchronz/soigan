library("foreign")
X11.options(type="Xlib") 

filename <- "baseline_hit_rate_serial";
baseline_hit_rate_serial <- read.octave(paste("serial/", filename, ".mat", sep=""))
baseline_hit_rate_serial <- baseline_hit_rate_serial[[1]]
max_K = length(baseline_hit_rate_serial[, 1, 1])
for(K in 2:max_K) {
  pdf(file=paste(filename, "-K", K, ".pdf", sep=""))
  last <- length(baseline_hit_rate_serial[K, , 1])
  plot(2:last, baseline_hit_rate_serial[K, 2:last, 1], type="l", col="black", ylim=c(0, 1), xlab="Iteration", ylab="Hit rate", lty=1, lwd=2)
  lines(2:last, baseline_hit_rate_serial[K, 2:last, 2], type="l", col="blue", pch=22, lty=2, lwd=2)
  lines(2:last, baseline_hit_rate_serial[K, 2:last, 3], type="l", col="red", pch=22, lty=3, lwd=2)
  title(main="Clustering Hit Rate", col.main="black", font.main=4)
  legend(x="bottomright", legend=c("All", "1-rate", "0-rate"), col=c("black", "blue", "red"), lty=1:3, lwd=2)
  dev.off()
}

# TODO plot total rates for various K for comparison

