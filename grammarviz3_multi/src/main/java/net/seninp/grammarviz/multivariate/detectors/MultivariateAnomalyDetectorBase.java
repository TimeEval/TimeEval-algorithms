package net.seninp.grammarviz.multivariate.detectors;

import net.seninp.grammarviz.anomaly.RRAImplementation;
import net.seninp.grammarviz.gi.logic.GrammarRuleRecord;
import net.seninp.grammarviz.gi.logic.GrammarRules;
import net.seninp.grammarviz.gi.logic.RuleInterval;
import net.seninp.grammarviz.multivariate.configuration.GrammarVizAnomalyParameters;
import net.seninp.grammarviz.multivariate.configuration.MultivariateStrategy;
import net.seninp.grammarviz.multivariate.configuration.OutputMode;
import net.seninp.grammarviz.multivariate.timeseries.MultivariateTimeSeries;
import net.seninp.jmotif.distance.EuclideanDistance;
import net.seninp.jmotif.sax.NumerosityReductionStrategy;
import net.seninp.jmotif.sax.TSProcessor;
import net.seninp.jmotif.sax.datastructure.SAXRecords;
import net.seninp.jmotif.sax.discord.DiscordRecord;
import net.seninp.jmotif.sax.discord.DiscordRecords;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public abstract class MultivariateAnomalyDetectorBase implements MultivariateAnomalyDetector {
    private final Logger LOGGER = LoggerFactory.getLogger(MultivariateAnomalyDetectorBase.class);
    private static final TSProcessor tp = new TSProcessor();
    private final EuclideanDistance ed = new EuclideanDistance();
    protected final GrammarVizAnomalyParameters params;
    protected final MultivariateTimeSeries ts;

    public MultivariateAnomalyDetectorBase(GrammarVizAnomalyParameters params) throws IOException {
        this.params = params;
        LOGGER.info("Reading data ...");
        this.ts = new MultivariateTimeSeries(params.IN_FILE);
        if (ts.isUnivariate()) {
            LOGGER.info("Found only 1 dimension, reject MultivariateStrategy");
            params.MULTIVARIATE_STRATEGY = MultivariateStrategy.SEPARATE_MAX;
        }
        LOGGER.info(ts.toString());
    }

    public abstract void detect();

    protected ArrayList<RuleInterval> transformGrammarRules(GrammarRules rules, int nDims) throws
            CloneNotSupportedException {
        ArrayList<RuleInterval> intervals = new ArrayList<>(rules.size() * 2);

        // populate all intervals with their frequency
        for (GrammarRuleRecord rule : rules) {
            if (0 == rule.ruleNumber()) {
                continue;
            }
            for (RuleInterval ri : rule.getRuleIntervals()) {
                RuleInterval i = (RuleInterval) ri.clone();
                i.setCoverage(rule.getRuleIntervals().size());
                i.setId(rule.ruleNumber());

                i.setStart(i.getStart() * nDims);
                i.setEnd(i.getEnd() * nDims);
                intervals.add(i);
            }
        }
        return intervals;
    }

    protected double[] getCoverageArray(List<RuleInterval> rules, int tsLength) {
        double[] coverageArray = new double[tsLength];
        for (RuleInterval coveredInterval : rules) {
            int startPos = coveredInterval.getStart();
            int endPos = coveredInterval.getEnd();
            for (int j = startPos; j < endPos; j++) {
                coverageArray[j] = coverageArray[j] + 1;
            }
        }
        return coverageArray;
    }

    protected void writeRuleDensity(double[] coverages, int nDims) throws Exception {
        writeResults(adjustScoreForMerged(reformatCoverage(nDims, coverages), nDims));
    }

    protected void writeRuleDensity(List<double[]> coverages, List<Integer> nDims) throws Exception {
        assert (coverages.size() == nDims.size());
        ArrayList<double[]> scores = new ArrayList<>(coverages.size());
        for (int i = 0; i < coverages.size(); i++) {
            scores.add(adjustScoreForMerged(reformatCoverage(nDims.get(i), coverages.get(i)), nDims.get(i)));
        }
        LOGGER.info("Write rule density results");
        writeResults(aggregateResults(scores));
    }

    protected double[] aggregateResults(List<double[]> results) {
        double[] aggregatedResults = new double[results.get(0).length];
        for (int i = 0; i < aggregatedResults.length; i++) {
            for (double[] anomalyScores : results) {
                aggregatedResults[i] = Math.max(anomalyScores[i], aggregatedResults[i]);
            }
        }
        return aggregatedResults;
    }

    protected double[] adjustScoreForMerged(double[] score, int nDims) {
        if (nDims == 1) return score;
        double[] lengthAdjustedScore = new double[score.length / nDims];
        for (int i = 0; i < lengthAdjustedScore.length; i++) {
            lengthAdjustedScore[i] = score[i * nDims];
        }
        return lengthAdjustedScore;
    }

    protected double[] reformatCoverage(int nDims, double[] coverageArray) {
        double[] updatedCoverage = coverageArray.clone();
        // flips coverage, shifts it up and zeros start & end
        double max_coverage = Double.MIN_VALUE;
        for (int i = 0; i < updatedCoverage.length; i++) {
            if (updatedCoverage[i] > max_coverage) {
                max_coverage = updatedCoverage[i];
            }
            updatedCoverage[i] *= -1;
        }
        for (int i = 0; i < updatedCoverage.length; i++) {
            if (i < (params.SAX_WINDOW_SIZE * nDims) || i > (updatedCoverage.length - params.SAX_WINDOW_SIZE * nDims)) {
                updatedCoverage[i] = 0;
            } else {
                updatedCoverage[i] += max_coverage;
            }
        }
        return updatedCoverage;
    }

    protected void writeResults(double[] scores) throws IOException {
        File file = new File(params.DISTANCE_FILENAME);
        boolean created_dir = file.getParentFile().mkdirs();
        if (created_dir)
            LOGGER.info("Created directories for output!");
        BufferedWriter bw = new BufferedWriter(new FileWriter(params.DISTANCE_FILENAME));
        for (double score : scores) {
            bw.write(score + "\n");
        }
        bw.close();
    }

    protected double[] calculateAnomalyScores(double[] ts, List<RuleInterval> intervals, double[] coverage) throws Exception {
        return calculateAnomalyScores(ts, intervals, coverage, 1);
    }

    protected double[] calculateAnomalyScores(double[] ts, List<RuleInterval> intervals, double[] coverage, int nDims) throws Exception {

        LOGGER.info("Found " + intervals.size() + " intervals covered by rules");
        // get the coverage array
        if (params.OUTPUT_MODE == OutputMode.RULE_DENSITY) {
            reformatCoverage(nDims, coverage);
            return coverage;
        }

        // look for zero-covered intervals and add those to the list
        List<RuleInterval> zeros = getZeroIntervals(coverage);
        if (zeros.size() > 0) {
            LOGGER.info(
                    "found " + zeros.size() + " intervals not covered by rules: " + intervalsToString(zeros));
            intervals.addAll(zeros);
        } else {
            LOGGER.info("the whole timeseries is covered by rule intervals ...");
        }

        // optional discord output
        if (params.OUTPUT_MODE == OutputMode.DISCORDS) {
            return calculateTopDiscords(ts, params.DISCORDS_NUM, params.SAX_NORM_THRESHOLD, intervals);
        }

        intervals.sort((c1, c2) -> {
            if (c1.getStart() > c2.getStart()) {
                return 1;
            } else if (c1.getStart() < c2.getStart()) {
                return -1;
            }
            return 0;
        });

        // now lets find all the distances to non-self match
        double[] distances = new double[ts.length];
        double[] widths = new double[ts.length];
        calculateScores(ts, intervals, distances, widths, nDims);
        return reverseWindowing(distances, widths);
    }

    protected double[] calculateTopDiscords(double[] ts, int discordsToReport,
                                            double normalizationThreshold, List<RuleInterval> intervals) throws Exception {
        DiscordRecords discords = RRAImplementation.series2RRAAnomalies(ts, discordsToReport, intervals,
                normalizationThreshold);
        double[] discordArray = new double[ts.length];
        for (DiscordRecord rule : discords) {
            for (int i = rule.getPosition(); i < rule.getPosition() + rule.getLength(); i++) {
                discordArray[i] = Math.max(discordArray[i], rule.getNNDistance());
            }
        }
        return discordArray;
    }

    protected void calculateScores(double[] ts,
                                   List<RuleInterval> intervals,
                                   double[] distances,
                                   double[] widths, int nDims) throws Exception {
        for (RuleInterval ri : intervals) {
            int ruleStart = ri.getStart();
            int ruleEnd = ruleStart + ri.getLength();
            int window = ruleEnd - ruleStart;

            double[] cw = tp.subseriesByCopy(ts, ruleStart, ruleStart + window);

            //if a window cannot find a non-self match, this becomes its distance and outputs a warning in the framework
            double cwNNDist = Double.MAX_VALUE;

            // this effectively finds the furthest hit
            for (int j = 0; j < ts.length - window - 1; j += nDims) {
                if (Math.abs(ruleStart - j) > window) {
                    double[] currentSubsequence = tp.subseriesByCopy(ts, j, j + window);
                    double dist = ed.distance(cw, currentSubsequence) / window;
                    if (dist < cwNNDist) {
                        cwNNDist = dist;
                    }
                }
            }
            //TODO fix problem where rules can have the same start
            distances[ruleStart] = cwNNDist;
            widths[ruleStart] = ri.getLength();
        }
    }

    protected double[] reverseWindowing(double[] distances, double[] widths) {
        int[] activeWindows = new int[distances.length];
        double[] resultScores = new double[distances.length];
        for (int i = 0; i < distances.length; i++) {
            for (int j = 0; j < widths[i]; j++) {
                resultScores[i + j] += distances[i];
                activeWindows[i + j]++;
            }
        }
        for (int i = 0; i < distances.length; i++) {
            if (activeWindows[i] > 0) {
                resultScores[i] /= activeWindows[i];
            }
        }
        return resultScores;
    }

    protected String intervalsToString(List<RuleInterval> zeros) {
        StringBuilder sb = new StringBuilder();
        for (RuleInterval i : zeros) {
            sb.append(i.toString()).append(",");
        }
        return sb.toString();
    }

    protected List<RuleInterval> getZeroIntervals(double[] coverageArray) {
        ArrayList<RuleInterval> res = new ArrayList<>();
        int start = -1;
        boolean inInterval = false;
        int intervalsCounter = -1;
        for (int i = 0; i < coverageArray.length; i++) {
            if (0 == coverageArray[i] && !inInterval) {
                start = i;
                inInterval = true;
            }
            if (coverageArray[i] > 0 && inInterval) {
                res.add(new RuleInterval(intervalsCounter, start, i, 0));
                inInterval = false;
                intervalsCounter--;
            }
        }
        return res;
    }

    protected SAXRecords mergeDimensions(List<SAXRecords> dimensions,
                                       int mergeLength,
                                       NumerosityReductionStrategy nrStrategy) {
        SAXRecords mergedRecords = new SAXRecords();
        String previousWord = "";
        for (int idx = 0; idx < mergeLength; idx++) {
            StringBuilder buffer = new StringBuilder();
            for (SAXRecords dimRecords : dimensions) {
                buffer.append(dimRecords.getByIndex(idx).getPayload());
            }
            String mergedWord = buffer.toString();
            if (nrStrategy != NumerosityReductionStrategy.EXACT || !mergedWord.equals(previousWord)) {
                mergedRecords.add(mergedWord.toCharArray(), idx);
                previousWord = mergedWord;
            }
        }
        return mergedRecords;
    }
}
