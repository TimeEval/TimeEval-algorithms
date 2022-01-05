#!/usr/bin/env Rscript

library(jsonlite)
library(PST)
library(TraMineR)
library(arules)
library(BBmisc)

split_into_subsequences = function(values, subsequence_count, subsequence_length) {
  subsequences <- matrix(0L, nrow=subsequence_count, ncol=subsequence_length)
  for(row in 1:nrow(subsequences)) {
    for(col in 1:ncol(subsequences)) {
      subsequences[row, col] <- values[row + col - 1]
    }
  }
  return(subsequences)
}

compute_similarity_scores = function(pst, subsequences, sim) {
  if (sim == "simo")
    sim <- "SIMo"
  else if (sim == "simn")
    sim <- "SIMn"
  similarity_scores <- predict(pst, subsequences, output=sim)
  return(similarity_scores)
}

compute_anomaly_scores_for_points = function(similarity_scores, values, subsequence_length) {
  anomaly_scores <- matrix(0L, nrow=length(values), ncol=3)
  for(i in 1:length(similarity_scores)){
    for(j in i:(i+subsequence_length-1)){
      anomaly_scores[j, 1] <- anomaly_scores[j, 1] + similarity_scores[i]
      anomaly_scores[j, 2] <- anomaly_scores[j, 2] + 1
    }
  }
  return(anomaly_scores[, 1] / anomaly_scores[, 2])
}

`%or%` = function(a, b) {
  cmp = function(a,b) if (
    identical(a, FALSE) || is.null(a) ||
    is.na(a) || is.nan(a) || length(a) == 0
  ) b else a

  if (length(a) > 1)
    mapply(cmp, a, b)
  else
    cmp(a, b)
}

args <- commandArgs(trailingOnly = TRUE)
if (is.null(args) || is.na(args) || is.nan(args) || length(args) == 0) {
  stop("No arguments specified!")
}

config <- fromJSON(args)
input <- config$dataInput %or% "/data/dataset.csv"
output <- config$dataOutput %or% "/results/anomaly_scores.csv"
# ignore modelInput and modelOutput, because they are not needed
executionType <- config$executionType %or% "execute"
window_size <- config$customParameters$window_size %or% 5
n <- config$customParameters$n %or% 1
n_min <- config$customParameters$n_min %or% 30
y_min <- config$customParameters$y_min %or% NULL
max_depth <- config$customParameters$max_depth %or% 4
n_bins <- config$customParameters$n_bins %or% 5
sim <- config$customParameters$sim %or% "SIMn"
sim <- tolower(sim)
random_state <- config$customParameters$random_state %or% 42

# Set random seed
set.seed(random_state)

# Check if the parameters are valid
if (sim != "simn" & sim != "simo") {
  message("ERROR: sim has to be SIMn or SIMo")
  quit(status=1)
}
if (window_size<=max_depth | window_size<=4) {
  message("WARN: window_size has to be greater than max(4, max_depth). Dynamically fixing it by setting window_size to ", max(5, max_depth+1))
  window_size <- max(5, max_depth+1)
}
if (executionType != "execute") {
  message("Training not required. Finished!")
  quit()
}

message("Reading data from ", input)
data <- read.csv(file=input)
values <- data[, 2] # Attention: 1-based indexing!

message("Discretize the time-series by frequency.")
values <- discretize(values, method="frequency", breaks=n_bins, labels=NULL)

message("Split input vector into matrix of subsequences")
subsequence_count <- length(values) - window_size + 1
subsequences <- split_into_subsequences(values, subsequence_count, window_size)

message("Transform the Subsquences into an sequence object")
sequences <- seqdef(subsequences)

message("Bulding the PST")
pst <- pstree(sequences, nmin=n_min, ymin=y_min, L=max_depth, lik=FALSE)

message("Compute the Similarity from all Sequences with respect to the PST")
similarity_scores <- compute_similarity_scores(pst, sequences, sim)

message("Revert and Normalize the similarity scores.")
similarity_scores <- normalize(similarity_scores, method="range", range=c(1,0), margin=2)

message("Build result")
anomaly_scores <- compute_anomaly_scores_for_points(similarity_scores, values, window_size)

message("Writing results to ", output)
write.table(anomaly_scores, file=output, sep=",", eol="\n", row.names = FALSE, col.names = FALSE, fileEncoding="UTF-8")
