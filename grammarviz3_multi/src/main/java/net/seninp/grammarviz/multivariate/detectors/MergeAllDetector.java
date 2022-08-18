package net.seninp.grammarviz.multivariate.detectors;

import net.seninp.grammarviz.gi.logic.GrammarRules;
import net.seninp.grammarviz.gi.logic.RuleInterval;
import net.seninp.grammarviz.multivariate.configuration.GrammarVizAnomalyParameters;
import net.seninp.grammarviz.multivariate.configuration.OutputMode;
import net.seninp.grammarviz.multivariate.timeseries.UnivariateTimeSeries;
import net.seninp.jmotif.sax.NumerosityReductionStrategy;
import net.seninp.jmotif.sax.SAXException;
import net.seninp.jmotif.sax.datastructure.SAXRecords;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static net.seninp.grammarviz.gi.sequitur.SequiturFactory.dicretizeTS;
import static net.seninp.grammarviz.gi.sequitur.SequiturFactory.discretized2SequiturRules;

public class MergeAllDetector extends MultivariateAnomalyDetectorBase {

    public MergeAllDetector(GrammarVizAnomalyParameters params) throws IOException {
        super(params);
    }

    @Override
    public void detect() {
        try {
            mergeAll();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void mergeAll() throws Exception {
        int nDims = ts.getWidth();
        int length = ts.getLength();
        GrammarRules rules = multiSeries2SequiturRules();
        // merge time series in the same order as the rules
        double[] mergedTimeSeries = new double[nDims * length];
        for (int j = 0; j < length; j++) {
            for (int dim = 0; dim < nDims; dim++) {
                mergedTimeSeries[j * nDims + dim] = ts.get(dim, j);
            }
        }
        ArrayList<RuleInterval> intervals = transformGrammarRules(rules, nDims);
        double[] coverageArray = getCoverageArray(intervals, mergedTimeSeries.length);
        writeRuleDensity(coverageArray, nDims);
        if (params.OUTPUT_MODE != OutputMode.RULE_DENSITY) {
            double[] anomalyScore = calculateAnomalyScores(mergedTimeSeries, intervals, coverageArray, nDims);
            double[] lengthAdjustedScore = adjustScoreForMerged(anomalyScore, nDims);
            writeResults(lengthAdjustedScore);
        }
    }

    private GrammarRules multiSeries2SequiturRules() throws SAXException {
        List<SAXRecords> saxFrequencyData = new ArrayList<>();
        for (UnivariateTimeSeries dimension : ts) {
            saxFrequencyData.add(dicretizeTS(dimension, params, NumerosityReductionStrategy.NONE));
        }
        SAXRecords mergedRecords = mergeDimensions(saxFrequencyData, ts.getLength() - params.SAX_WINDOW_SIZE, params.SAX_NR_STRATEGY);
        return discretized2SequiturRules(mergedRecords, params.SAX_WINDOW_SIZE, params.SAX_PAA_SIZE, ts.getLength());
    }
}
