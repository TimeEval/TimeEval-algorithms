package net.seninp.grammarviz;

import com.beust.jcommander.JCommander;
import net.seninp.grammarviz.multivariate.configuration.GrammarVizAnomalyParameters;
import net.seninp.grammarviz.multivariate.configuration.TimeEvalArguments;
import net.seninp.grammarviz.multivariate.detectors.MergeAllDetector;
import net.seninp.grammarviz.multivariate.detectors.MergeClusteredDetector;
import net.seninp.grammarviz.multivariate.detectors.MultivariateAnomalyDetector;
import net.seninp.grammarviz.multivariate.detectors.SeparateMaxDetector;
import net.seninp.jmotif.distance.EuclideanDistance;
import net.seninp.jmotif.sax.TSProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class GrammarVizAnomaly {

    private static final String CR = "\n";
    private static final TSProcessor tp = new TSProcessor();
    private static final EuclideanDistance ed = new EuclideanDistance();
    private static final Logger LOGGER = LoggerFactory.getLogger(GrammarVizAnomaly.class);
    public static final GrammarVizAnomalyParameters params = new GrammarVizAnomalyParameters();

    public static void main(String[] args) throws Exception {
        JCommander jct = new JCommander(params, args);
        if (args.length == 0) {
            jct.usage();
            System.exit(1);
        }
        // TimeEval integration
        if (args.length == 1 && args[0].startsWith("{")) {
            TimeEvalArguments teParams = TimeEvalArguments.fromJson(args[0]);
            teParams.overwriteParameters(params);
            LOGGER.info("timeeval integration parsing");
        }
        printParameters();

        MultivariateAnomalyDetector detector;
        switch (params.MULTIVARIATE_STRATEGY) {
            case MERGE_ALL:
                detector = new MergeAllDetector(params);
                break;
            case MERGE_CLUSTERED:
                detector = new MergeClusteredDetector(params);
                break;
            default:
                detector = new SeparateMaxDetector(params);
                break;
        }
        detector.detect();
    }

    private static void printParameters() {
        String sb = CR + "GrammarViz2 CLI anomaly discovery" + CR + "parameters:" + CR +
                " input file:                  " + params.IN_FILE + CR +
                " output files prefix:         " + params.OUT_FILE + CR +
                " Algorithm implementation:    " + params.ALGORITHM + CR +
                " Num. of discords to report:  " + params.DISCORDS_NUM + CR +
                " Output strategy:             " + params.OUTPUT_MODE + CR +
                " Multivariate strategy:       " + params.MULTIVARIATE_STRATEGY + CR +
                " SAX sliding window size:     " + params.SAX_WINDOW_SIZE + CR +
                " SAX PAA size:                " + params.SAX_PAA_SIZE + CR +
                " SAX alphabet size:           " + params.SAX_ALPHABET_SIZE + CR +
                " SAX numerosity reduction:    " + params.SAX_NR_STRATEGY + CR +
                " SAX normalization threshold: " + params.SAX_NORM_THRESHOLD + CR;
        System.out.println(sb);
    }
}
