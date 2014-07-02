#!/bin/bash

if [ $# == 0 ]; then
  experiments=`ls "results/"`
else
  experiments=$@
fi
  echo $experiments
for experiment in $experiments
do
  echo "Current $experiment"
  # transform results for plotting in R
  rm *.mat
  cp results/$experiment/experimentResultsSerial.mat .
  ls *.mat

  # Create the directory for the serial plots.
  if [ ! -d "plotting/serial" ]; then
    mkdir plotting/serial
  fi
  rm plotting/serial/*

  # Baseline
  octave plotBaseline.m
  mv baseline_hit_rate_serial.mat plotting/serial/
  mv baseline_prediction_serial.mat plotting/serial/
  mv baseline_training_serial.mat plotting/serial/
  cd plotting
  R --file=plotBaseline.r
  cd ..
  # Bernoulli
  octave plotBernoulli.m
  mv bernoulli_hit_rate.mat plotting/serial/
  mv bernoulli_prediction_serial.mat plotting/serial/
  mv bernoulli_training_serial.mat plotting/serial/
  cd plotting
  R --file=plotBernoulli.r
  cd ..
  # Prob model
  octave plotProbModel.m
  mv prob_model_hit_rate_serial.mat plotting/serial/
  mv prob_model_prediction_serial.mat plotting/serial/
  mv prob_model_training_serial.mat plotting/serial/
  cd plotting
  R --file=plotProbModel.r
  cd ..
  # SVM
  octave plotSvm.m
  mv svm_hit_rate.mat plotting/serial/
  mv svm_prediction_serial.mat plotting/serial/
  mv svm_training_serial.mat plotting/serial/
  cd plotting
  R --file=plotSvm.r
  cd ..

  # Comparative plots
  cd plotting
  R --file=plotCombined.r

  if [ ! -d "$experiment" ]; then
    mkdir $experiment
  fi
  rm $experiment/*
  mv *.pdf $experiment

  cd ..
  echo "Done with $experiment"
done

echo "Done $experiments"

