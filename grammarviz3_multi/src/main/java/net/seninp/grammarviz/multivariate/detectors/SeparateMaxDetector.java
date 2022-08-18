package net.seninp.grammarviz.multivariate.detectors;

import net.seninp.grammarviz.gi.logic.GrammarRules;
import net.seninp.grammarviz.gi.logic.RuleInterval;
import net.seninp.grammarviz.gi.sequitur.SequiturFactory;
import net.seninp.grammarviz.multivariate.configuration.GrammarVizAnomalyParameters;
import net.seninp.grammarviz.multivariate.configuration.OutputMode;
import net.seninp.grammarviz.multivariate.timeseries.UnivariateTimeSeries;
import net.seninp.jmotif.sax.datastructure.SAXRecords;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class SeparateMaxDetector extends MultivariateAnomalyDetectorBase {

    public SeparateMaxDetector(GrammarVizAnomalyParameters params) throws IOException {
        super(params);
    }

    @Override
    public void detect() {
        try {
            multivariateSeparately();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void multivariateSeparately() throws Exception {
        List<double[]> all_scores = new ArrayList<>();
        List<List<RuleInterval>> intervals = new ArrayList<>();
        List<double[]> coverages = new ArrayList<>();

        for (UnivariateTimeSeries dimension : ts) {
            SAXRecords discretizedDimension = SequiturFactory.dicretizeTS(
                    dimension,
                    params,
                    params.SAX_NR_STRATEGY);

            GrammarRules rules = SequiturFactory.discretized2SequiturRules(
                    discretizedDimension,
                    params.SAX_WINDOW_SIZE,
                    params.SAX_PAA_SIZE,
                    dimension.getLength());

            ArrayList<RuleInterval> current_intervals = transformGrammarRules(rules, 1);
            intervals.add(current_intervals);
            coverages.add(getCoverageArray(current_intervals, dimension.getLength()));

        }
        writeRuleDensity(coverages, Collections.nCopies(coverages.size(), 1));
        if (params.OUTPUT_MODE != OutputMode.RULE_DENSITY) {
            for (int i = 0; i < intervals.size(); i++) {
                all_scores.add(calculateAnomalyScores(ts.get(i).getRawData(), intervals.get(i), coverages.get(i)));
            }
            writeResults(aggregateResults(all_scores));
        }

    }
}
