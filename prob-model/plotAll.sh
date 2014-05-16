#!/bin/bash

experiments=`ls "results/"`
for experiment in $experiments
do
  echo "Current $experiment"
  # transform results for plotting in R
  rm *.mat
  cp results/$experiment/experimentResultsSerial.mat .
  cp results/$experiment/experimentResultsParallel.mat .
  ls *.mat
  octave plotResults.m

  # serial plots
  if [ ! -d "plotting/serial" ]; then
    mkdir plotting/serial
  fi
  rm plotting/serial/*
  mv baseline_hit_rate_serial.mat plotting/serial/
  mv baseline_prediction_serial.mat plotting/serial/
  mv baseline_training_serial.mat plotting/serial/
  mv svm_hit_rate.mat plotting/serial/
  mv svm_prediction_serial.mat plotting/serial/
  mv svm_training_serial.mat plotting/serial/

  ## parallel plots
  #if [ ! -d "plotting/parallel" ]; then
  #  mkdir plotting/parallel
  #fi
  #rm plotting/parallel/*
  #mv baseline_accuracy.mat plotting/parallel
  #mv baseline_learning.mat plotting/parallel
  #mv baseline_prediction.mat plotting/parallel
  #mv svm_accuracy.mat plotting/parallel
  #mv svm_learning.mat plotting/parallel
  #mv svm_prediction.mat plotting/parallel

  cd plotting
  R --file=plotResults.r

  if [ ! -d "$experiment" ]; then
    mkdir $experiment
  fi
  rm $experiment/*
  mv *.pdf $experiment

  cd ..
  echo "Done with $experiment"
done

echo "Done All"

