package net.seninp.grammarviz.cli;

import com.beust.jcommander.JCommander;
import net.seninp.gi.logic.GrammarRules;
import net.seninp.gi.rulepruner.RulePrunerFactory;
import net.seninp.gi.sequitur.SAXRule;
import net.seninp.gi.sequitur.SequiturFactory;
import net.seninp.jmotif.sax.SAXException;
import net.seninp.jmotif.sax.SAXProcessor;
import net.seninp.jmotif.sax.TSProcessor;
import net.seninp.jmotif.sax.alphabet.NormalAlphabet;
import net.seninp.jmotif.sax.datastructure.SAXRecords;
import net.seninp.jmotif.sax.parallel.ParallelSAXImplementation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * CLI for SAX + Sequitur rules production and pruning.
 */
public class TS2SequiturGrammar {

    private static Logger consoleLogger = LoggerFactory.getLogger(TS2SequiturGrammar.class);
    private static final String NEWLINE = System.lineSeparator();

    private static TSProcessor tp = new TSProcessor();
    private static NormalAlphabet na = new NormalAlphabet();
    private static SAXProcessor sp = new SAXProcessor();
    private static ParallelSAXImplementation psax = new ParallelSAXImplementation();

    public static void main(String[] args) throws Exception {
        parseArgs(args);

        double[] series = readTimeSeries();
        SAXRecords saxData = discretize(series);
        String str = saxData.getSAXString(" ");
        GrammarRules rules = inferGrammarRules(series, saxData, str);

        if (TS2GrammarParameters.PRUNE_RULES) {
            consoleLogger.info("Pruning rules ...");
            rules = RulePrunerFactory.performPruning(series, rules);
        }

        consoleLogger.info("Producing the output ...");
        new RulesWriter(TS2GrammarParameters.OUT_FILE)
                .write(rules);
    }

    private static void parseArgs(String[] args) {
        TS2GrammarParameters params = new TS2GrammarParameters();
        JCommander jct = new JCommander(params, args);

        if (0 == args.length) {
            jct.usage();
        } else {
            logParams();
        }
    }

    private static void logParams() {
        String output = "GrammarViz2 CLI converter v.1" + NEWLINE +
                "parameters:" + NEWLINE +
                "  input file:                  " + TS2GrammarParameters.IN_FILE + NEWLINE +
                "  output file:                 " + TS2GrammarParameters.OUT_FILE + NEWLINE +
                "  SAX sliding window size:     " + TS2GrammarParameters.SAX_WINDOW_SIZE + NEWLINE +
                "  SAX PAA size:                " + TS2GrammarParameters.SAX_PAA_SIZE + NEWLINE +
                "  SAX alphabet size:           " + TS2GrammarParameters.SAX_ALPHABET_SIZE + NEWLINE +
                "  SAX numerosity reduction:    " + TS2GrammarParameters.SAX_NR_STRATEGY + NEWLINE +
                "  SAX normalization threshold: " + TS2GrammarParameters.SAX_NORM_THRESHOLD + NEWLINE +
                "  Pruning rules:               " + TS2GrammarParameters.PRUNE_RULES + NEWLINE +
                NEWLINE;

        consoleLogger.info(output);
    }

    private static GrammarRules inferGrammarRules(double[] series, SAXRecords saxData, String str) throws Exception {
        consoleLogger.info("Inferring Sequitur grammar ...");
        SAXRule grammar = SequiturFactory.runSequitur(str);

        consoleLogger.info("Collecting stats ...");
        GrammarRules rules = grammar.toGrammarRulesData();

        SequiturFactory.updateRuleIntervals(rules, saxData, true, series,
                                            TS2GrammarParameters.SAX_WINDOW_SIZE, TS2GrammarParameters.SAX_PAA_SIZE);
        return rules;
    }

    private static SAXRecords discretize(double[] series) throws SAXException {
        consoleLogger.info("Performing SAX conversion ...");
        if (TS2GrammarParameters.NUM_WORKERS <= 1) {
            return sp.ts2saxViaWindow(series, TS2GrammarParameters.SAX_WINDOW_SIZE,
                                      TS2GrammarParameters.SAX_PAA_SIZE, na.getCuts(TS2GrammarParameters.SAX_ALPHABET_SIZE),
                                      TS2GrammarParameters.SAX_NR_STRATEGY, TS2GrammarParameters.SAX_NORM_THRESHOLD);
        } else {
            return psax.process(series, TS2GrammarParameters.NUM_WORKERS, TS2GrammarParameters.SAX_WINDOW_SIZE,
                                      TS2GrammarParameters.SAX_PAA_SIZE, TS2GrammarParameters.SAX_ALPHABET_SIZE,
                                      TS2GrammarParameters.SAX_NR_STRATEGY, TS2GrammarParameters.SAX_NORM_THRESHOLD);
        }
    }

    private static double[] readTimeSeries() throws SAXException, IOException {
        consoleLogger.info("Reading data ...");
        double[] series = tp.readTS(TS2GrammarParameters.IN_FILE, 0);
        consoleLogger.info("read " + series.length + " points from " + TS2GrammarParameters.IN_FILE);
        return series;
    }
}
