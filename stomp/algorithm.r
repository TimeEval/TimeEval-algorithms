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
n_jobs <- config$customParameters$n_jobs %or% 1
random_state <- config$customParameters$random_state %or% 42
use_column_index <- config$customParameters$use_column_index %or% 0
# align index to R-indexing, which is 1-based
use_column_index <- use_column_index + 1

# Set random seed
set.seed(random_state)

# check parameters
if (window_size < 4) {
  message("WARN: window_size must be at least 4. Dynamically fixing it by setting window_size to 4")
  window_size <- 4
}

if (verbose > 1) {
    message("-- Configuration ------------")
    message("executionType=", executionType)
    message("window_Size=", window_size)
    message("exclusion_zone=", exclusion_zone)
    message("verbose=", verbose)
    message("n_jobs=", n_jobs)
    message("-----------------------------")
}

if (executionType != "execute") {
    message("Training not required. Finished!")
    quit()
}


message("Reading data from ", input)
data <- read.csv(file=input)

max_column_index <- ncol(data) - 2
if (use_column_index > max_column_index) {
    message("Selected column index ",
        use_column_index,
        " is out of bounds (max index = ",
        max_column_index,
        ")! Using last channel!"
    )
    use_column_index <- max_column_index
}
# jump over index column (timestamp)
use_column_index <- use_column_index + 1
values = data[, use_column_index] # Attention: 1-based indexing!

if (n_jobs <= 1) {
    stomp_mp <- stomp(values, window_size=window_size, exclusion_zone=exclusion_zone, verbose=verbose)
} else {
    stomp_mp <- stomp_par(values, window_size=window_size, exclusion_zone=exclusion_zone, verbose=verbose, n_workers=n_jobs)
}
result <- stomp_mp$mp[,1]

message("Writing results to ", output)
write.table(result, file=output, sep=",", eol="\n", row.names = FALSE, col.names = FALSE, fileEncoding="UTF-8")
