package net.seninp.grammarviz.multivariate.detectors;

import com.apporiented.algorithm.clustering.Cluster;
import com.apporiented.algorithm.clustering.ClusteringAlgorithm;
import com.apporiented.algorithm.clustering.CompleteLinkageStrategy;
import com.apporiented.algorithm.clustering.PDistClusteringAlgorithm;
import net.seninp.grammarviz.gi.logic.GrammarRules;
import net.seninp.grammarviz.gi.logic.RuleInterval;
import net.seninp.grammarviz.gi.sequitur.SequiturFactory;
import net.seninp.grammarviz.multivariate.configuration.GrammarVizAnomalyParameters;
import net.seninp.grammarviz.multivariate.configuration.OutputMode;
import net.seninp.grammarviz.multivariate.metrics.Compressibility;
import net.seninp.jmotif.sax.NumerosityReductionStrategy;
import net.seninp.jmotif.sax.SAXException;
import net.seninp.jmotif.sax.datastructure.SAXRecords;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class MergeClusteredDetector extends MultivariateAnomalyDetectorBase {
    private final Logger LOGGER = LoggerFactory.getLogger(MergeClusteredDetector.class);
    private final double correlationThreshold;

    public MergeClusteredDetector(GrammarVizAnomalyParameters params) throws IOException {
        super(params);
        correlationThreshold = 0.7;
    }

    public MergeClusteredDetector(GrammarVizAnomalyParameters params, double threshold) throws IOException {
        super(params);
        correlationThreshold = threshold;
    }

    @Override
    public void detect() {
        try {
            clusterCorrelated();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void clusterCorrelated() throws Exception {
        List<SAXRecords> saxFrequencyData = saxTimeSeries();
        List<List<Integer>> mergedDimensions = findCorrelateDimensions(saxFrequencyData);
        List<double[]> mergedTimeSeriesCollection = mergeCorrelatedTimeSeries(mergedDimensions);
        saxFrequencyData = mergeCorrelatedSaxRecords(saxFrequencyData, mergedDimensions);
        generateClusteredAnomalyScores(mergedTimeSeriesCollection, saxFrequencyData, mergedDimensions);
    }


    private ArrayList<SAXRecords> saxTimeSeries() throws SAXException {
        // potentially modifies series in place
        ArrayList<SAXRecords> saxFrequencyData = new ArrayList<>();
        for (int i = 0; i < ts.getWidth(); i++) {
            SAXRecords saxRecords = SequiturFactory.dicretizeTS(ts.get(i),
                    params,
                    NumerosityReductionStrategy.NONE);
            if (saxRecords.getRecords().size() > 1) {
                saxFrequencyData.add(saxRecords);
            } else {
                ts.remove(i);
                i--;
            }
        }
        return saxFrequencyData;
    }

    private List<SAXRecords> mergeCorrelatedSaxRecords(List<SAXRecords> saxFrequencyData,
                                                       List<List<Integer>> mergedDimensions) {
        List<SAXRecords> newSaxRecords = new ArrayList<>(mergedDimensions.size());
        for (List<Integer> currentCluster : mergedDimensions) {
            newSaxRecords.add(mergeDimensions(
                    currentCluster.stream().map(saxFrequencyData::get).collect(Collectors.toList()),
                    ts.getLength() - params.SAX_WINDOW_SIZE,
                    params.SAX_NR_STRATEGY));
        }
        return newSaxRecords;
    }

    private List<List<Integer>> findCorrelateDimensions(List<SAXRecords> saxFrequencyData) {
        // potentially modifies saxFrequencyData in place
        double[][] distances = calculateDistanceMatrix(saxFrequencyData);
        String[] dimensionTags = IntStream.range(0, ts.getWidth()).mapToObj(String::valueOf).toArray(String[]::new);
        System.out.println(distances[0].length);
        System.out.println(dimensionTags.length);
        System.out.println(Arrays.toString(dimensionTags));
        ClusteringAlgorithm algorithm = new PDistClusteringAlgorithm();
        Cluster cluster = algorithm.performClustering(distances, dimensionTags, new CompleteLinkageStrategy());
        return findMergedDimensions(cluster);
    }

    private List<List<Integer>> findMergedDimensions(Cluster mergedClusters) {
        List<List<Integer>> mergedDimensions = new ArrayList<>();
        LinkedList<Cluster> clustersToSearch = new LinkedList<>();
        clustersToSearch.add(mergedClusters);
        while (!clustersToSearch.isEmpty()) {
            Cluster currentCluster = clustersToSearch.pop();
            if (-1 * currentCluster.getDistance().getDistance() >= this.correlationThreshold
                    || currentCluster.getChildren() == null
                    || currentCluster.getChildren().isEmpty()) {
                mergedDimensions.add(getClusterDims(currentCluster));
            } else {
                clustersToSearch.addAll(currentCluster.getChildren());
            }
        }
        return mergedDimensions;
    }

    private List<Integer> getClusterDims(Cluster c) {
        List<Integer> mergedDimensions = new ArrayList<>();
        for (String dimension : c.getLeafNames()) {
            mergedDimensions.add(Integer.valueOf(dimension));
        }
        return mergedDimensions;
    }

    private double[][] calculateDistanceMatrix(List<SAXRecords> saxFrequencyData) {
        int n = ts.getWidth();
        double[][] distances = new double[1][n * (n - 1) / 2];
        int idx = 0;
        for (int i = 0; i < saxFrequencyData.size(); i++) {
            SAXRecords currentDim = saxFrequencyData.get(i);
            double currentDimCompressibility = Compressibility.getCompressibility(
                    currentDim.getRecords().size(), ts.getLength(), params.SAX_WINDOW_SIZE);
            for (int j = i + 1; j < saxFrequencyData.size(); j++) {
                SAXRecords comparedDim = saxFrequencyData.get(j);
                double comparedDimCompressibility = Compressibility.getCompressibility(comparedDim.getRecords().size(),
                        ts.getLength(), params.SAX_WINDOW_SIZE);

                SAXRecords mergedDims = mergeDimensions(Arrays.asList(currentDim, comparedDim),
                        ts.getLength() - params.SAX_WINDOW_SIZE,
                        params.SAX_NR_STRATEGY);

                double mergedDimsCompressibility = Compressibility.getCompressibility(mergedDims.getRecords().size(),
                        ts.getLength(), params.SAX_WINDOW_SIZE);
                double jointCompressibility = Compressibility.getJointCompressibility(currentDimCompressibility,
                        comparedDimCompressibility, mergedDimsCompressibility);
                distances[0][idx] = -1 * jointCompressibility;
                idx++;
            }
        }
        return distances;
    }


    private SAXRecords reduceNumerosity(SAXRecords timeSeries, int tsLength, NumerosityReductionStrategy nrStrategy) {
        SAXRecords reducedRecord = new SAXRecords();
        String previousWord = "";
        for (int idx = 0; idx < tsLength; idx++) {
            String word = new String(timeSeries.getByIndex(idx).getPayload());
            if (nrStrategy != NumerosityReductionStrategy.EXACT || !word.equals(previousWord)) {
                reducedRecord.add(word.toCharArray(), idx);
                previousWord = word;
            }
        }
        return reducedRecord;
    }

    private List<double[]> mergeCorrelatedTimeSeries(List<List<Integer>> whichDimensionsWhereMerged) {
        List<double[]> mergedTimeSeriesCollection = new ArrayList<>();
        for (List<Integer> dimension : whichDimensionsWhereMerged) {
            if (dimension.size() > 1) {
                double[] mergedTimeSeries = new double[ts.getLength() * dimension.size()];
                for (int j = 0; j < ts.getLength(); j++) {
                    for (int dim = 0; dim < dimension.size(); dim++) {
                        mergedTimeSeries[j * dimension.size() + dim] = ts.get(dimension.get(dim), j);
                    }
                }
                mergedTimeSeriesCollection.add(mergedTimeSeries);
            } else {
                mergedTimeSeriesCollection.add(ts.get(dimension.get(0)).getRawData());
            }
        }
        return mergedTimeSeriesCollection;
    }

    private void generateClusteredAnomalyScores(List<double[]> mergedTimeSeriesCollection,
                                                List<SAXRecords> saxFrequencyData,
                                                List<List<Integer>> whichDimensionsWhereMerged) throws Exception {
        List<List<RuleInterval>> intervals = new ArrayList<>();
        List<double[]> coverages = new ArrayList<>();
        List<Integer> nDims = new ArrayList<>();

        for (int i = 0; i < saxFrequencyData.size(); i++) {
            GrammarRules rules = SequiturFactory.discretized2SequiturRules(saxFrequencyData.get(i),
                    params.SAX_WINDOW_SIZE,
                    params.SAX_PAA_SIZE,
                    ts.getLength());
            if (whichDimensionsWhereMerged.get(i).size() > 1) {
                LOGGER.info("Merged " + whichDimensionsWhereMerged.get(i).size() + " Dimensions");
            }
            nDims.add(whichDimensionsWhereMerged.get(i).size());
            intervals.add(transformGrammarRules(rules, nDims.get(i)));
            coverages.add(getCoverageArray(intervals.get(i), mergedTimeSeriesCollection.get(i).length));
        }
        writeRuleDensity(coverages, nDims);
        ArrayList<double[]> all_scores = new ArrayList<>();
        if (params.OUTPUT_MODE != OutputMode.RULE_DENSITY) {
            for (int i = 0; i < intervals.size(); i++) {
                double[] anomalyScore = calculateAnomalyScores(mergedTimeSeriesCollection.get(i), intervals.get(i), coverages.get(i), nDims.get(i));

                double[] lengthAdjustedScore = adjustScoreForMerged(anomalyScore, nDims.get(i));
                all_scores.add(lengthAdjustedScore);
            }
            LOGGER.info("Write final clustered results.");
            writeResults(aggregateResults(all_scores));
        }
    }

}
