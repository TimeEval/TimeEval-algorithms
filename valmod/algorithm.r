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

config <- fromJSON(args)
input <- config$dataInput %or% "/data/dataset.csv"
output <- config$dataOutput %or% "/results/anomaly_scores.csv"
# ignore modelInput and modelOutput, because they are not needed
executionType <- config$executionType %or% "execute"
window_min <- config$customParameters$min_anomaly_window_size %or% 30
window_max <- config$customParameters$max_anomaly_window_size %or% 40
# window size must be at least 4
window_min <- max(window_min, 4)
window_max <- max(window_min + 1, window_max)
heap_size <- config$customParameters$heap_size %or% 50
exclusion_zone <- config$customParameters$exclusion_zone %or% 0.5
verbose <- config$customParameters$verbose %or% 1
random_state <- config$customParameters$random_state %or% 42

# Set random seed
set.seed(random_state)

if(verbose > 1) {
    message("-- Configuration ------------")
    message("executionType=", executionType)
    message("window_min=", window_min)
    message("window_max=", window_max)
    message("heap_size=", heap_size)
    message("exclusion_zone=", exclusion_zone)
    message("verbose=", verbose)
    message("-----------------------------")
}

if (executionType != "execute") {
    message("Training not required. Finished!")
    quit()
}


message("Reading data from ", input)
data <- read.csv(file=input)
values = data[, 2] # Attention: 1-based indexing!

valmod_mp <- valmod(values,window_min=window_min, window_max=window_max, heap_size=heap_size, exclusion_zone=exclusion_zone, verbose=verbose)
result <- valmod_mp$mp[,1]

message("Writing results to ", output)
write.table(result, file=output, sep=",", eol="\n", row.names = FALSE, col.names = FALSE, fileEncoding="UTF-8")
