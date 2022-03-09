#!/usr/bin/env Rscript

library(jsonlite)
library(stream)
library(BBmisc)

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

preprocess = function(values, subsequence_length, subsequence_count, dimensionality) {
    subsequences <- array(0L, dim=c(subsequence_count, subsequence_length, dimensionality))
    values <- data.matrix(values)
    for(row in 1:subsequence_count) {
        for(col in 1:subsequence_length) {
            subsequences[row, col,] <- values[row + col - 1,]
        }
    }
    dim(subsequences) <- c(subsequence_count, subsequence_length * dimensionality)
    return(subsequences)
}

compute_anomaly_scores_for_sequences = function(macro_clusters, cluster_centers, micro_clusters, df, metric, weights) {
    anomaly_scores <- numeric(nrow(df))
    for (i in 1:nrow(df)) {
        # If the micro cluster of a point is not part of a macro cluster it will get the maximum anomaly score.
        # Therefore we set it to -1 first, and replace all -1's later.
        # Otherwise the negated weight of the macro cluster the point is in will be used as the anomaly score.
        if (is.na(macro_clusters[micro_clusters[i]])) {
            anomaly_scores[i] <- -1
        } else {
            # The commented code can be used to also include the distance from the point to the macro cluster center into the score.
            # X <- rbind(df[i, ], setNames(cluster_centers[macro_clusters[micro_clusters[i]],], names(df[i, ])))
            # anomaly_scores[i] <- dist(X, metric=metric) * -weights[macro_clusters[micro_clusters[i]]]
            anomaly_scores[i] <- -weights[macro_clusters[micro_clusters[i]]]
        }
    }
    max_score <- max(anomaly_scores)
    # Now replace all -1 with a higher score than the max anomaly score.
    anomaly_scores <- replace(anomaly_scores, anomaly_scores==-1, max_score+1)
    return(anomaly_scores)
}

compute_anomaly_scores_for_points = function(subsequence_anomaly_scores, subsequence_count, subsequence_length){
    anomaly_scores <- matrix(0L, nrow=(subsequence_count+subsequence_length-1), ncol=2)
    for(i in 1:subsequence_count){
        for(j in i:(i+subsequence_length-1)){
            anomaly_scores[j, 1] <- anomaly_scores[j, 1] + subsequence_anomaly_scores[i]
            anomaly_scores[j, 2] <- anomaly_scores[j, 2] + 1
    }
  }
  return(anomaly_scores[, 1] / anomaly_scores[, 2])
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
r <- config$customParameters$radius %or% .1
lambda <- config$customParameters$lambda %or% 0.001
metric <- config$customParameters$distance_metric %or% "Euclidean"
shared_density <- config$customParameters$shared_density %or% TRUE
alpha <- config$customParameters$alpha %or% 0.3
k <- config$customParameters$n_clusters %or% 0
minweight <- config$customParameters$min_weight %or% .0
subsequence_length <- config$customParameters$window_size %or% 20
random_state <- config$customParameters$random_state %or% 42

# Set random seed
set.seed(random_state)

if (executionType != "execute") {
    message("Training not required. Finished!")
    quit()
}

message("Reading data from ", input)
data <- read.csv(file=input)
values <- data[, 2:(ncol(data) - 1)] # Attention: 1-based indexing!

# normalize values into a range between 0 and 1
values <- normalize(values, method="range", range(0,1), margin=2L)

n <- nrow(values) %or% length(values)
subsequence_count <- (n - subsequence_length + 1)
dimensionality <- ncol(data) - 2

subsequences <- preprocess(values, subsequence_length, subsequence_count, dimensionality)
subsequences <- as.data.frame(subsequences)

message("Converting the data into a datastream object")
data_stream <- DSD_Memory(subsequences)

message("Parameters: window_count=", subsequence_count, ", window_length=", subsequence_length, ", k=", k, ", alpha=", alpha)

message("Initialize the algorithm with the correct parameters.")
# CM is fixed to 0 because we want every point to be assigned to micro cluster (even if at worst that cluster only contains of that point).
# Gaptime is set to the subsequence_count because we never want to have weak micro cluster removed.
dbstream <- DSC_DBSTREAM(r=r, lambda=lambda, gaptime=subsequence_count, Cm=0, metric=metric, shared_density=shared_density, alpha=alpha, minweight=minweight, k=k)

message("Apply the algorithm to all the point in the stream.")
update(dbstream, data_stream, n=subsequence_count, assignments=TRUE)

message("Get the cluster assignments.")
micro_clusters <- get_cluster_assignments(dbstream)
message("Get macro clustering", dbstream$RObj$get_macro_clustering())
message("Macros", dbstream$macro$macro)
message("Assignments", dbstream$macro$macro$microToMacro)
macro_clusters <- microToMacro(dbstream)

message("Get the macro cluster centers")
cluster_centers <- get_centers(dbstream, type="macro")

message("Compute the anomaly Scores for subsequences")
weights <- get_weights(dbstream, type='macro')
subsequence_anomaly_scores <- compute_anomaly_scores_for_sequences(macro_clusters, cluster_centers, micro_clusters, subsequences, tolower(metric), weights)

message("Compute the anomaly scores for points")
anomaly_scores <- compute_anomaly_scores_for_points(subsequence_anomaly_scores, subsequence_count, subsequence_length)

message("Writing results to ", output)
write.table(anomaly_scores, file=output, sep=",", eol="\n", row.names = FALSE, col.names = FALSE, fileEncoding="UTF-8")
