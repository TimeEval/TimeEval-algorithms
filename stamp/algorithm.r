#!/usr/bin/env Rscript

library(tsmp)
library(jsonlite)

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

config = fromJSON(args)
input <- config$dataInput %or% "/data/dataset.csv"
output <- config$dataOutput %or% "/results/anomaly_scores.csv"
# ignore modelInput and modelOutput, because they are not needed
executionType <- config$executionType %or% "execute"
window_size <- config$customParameters$anomaly_window_size %or% 30
exclusion_zone <- config$customParameters$exclusion_zone %or% 1/2
verbose <- config$customParameters$verbose %or% 1
s_size <- config$customParameters$s_size %or% Inf
n_jobs <- config$customParameters$n_jobs %or% 1
random_state <- config$customParameters$random_state %or% 42

# Set random seed
set.seed(random_state)

# check parameters
if (window_size < 4) {
  message("WARN: window_size must be at least 4. Dynamically fixing it by setting window_size to 4")
  window_size <- 4
}

if(verbose > 1) {
    message("-- Configuration ------------")
    message("executionType=", executionType)
    message("window_size=", window_size)
    message("exclusion_zone=", exclusion_zone)
    message("verbose=", verbose)
    message("s_size=", s_size)
    message("n_jobs=", n_jobs)
    message("-----------------------------")
}

if (executionType != "execute") {
    message("Training not required. Finished!")
    quit()
}

message("Reading data from ", input)
data <- read.csv(file=input)
values = data[, 2] # Attention: 1-based indexing!

if (n_jobs <= 1) {
    stamp_mp <- stamp(values, window_size=window_size, exclusion_zone=exclusion_zone, verbose=verbose, s_size=s_size)
} else {
    stamp_mp <- stamp_par(values, window_size=window_size, exclusion_zone=exclusion_zone, verbose=verbose, s_size=s_size, n_workers=n_jobs)
}
result <- stamp_mp$mp[,1]

message("Writing results to ", output)
write.table(result, file=output, sep=",", eol="\n", row.names = FALSE, col.names = FALSE, fileEncoding="UTF-8")
