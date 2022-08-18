package net.seninp.grammarviz.gi.sequitur;

import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;
import java.util.concurrent.atomic.AtomicInteger;

import net.seninp.grammarviz.multivariate.configuration.GrammarVizAnomalyParameters;
import net.seninp.grammarviz.multivariate.timeseries.MultivariateTimeSeries;
import net.seninp.grammarviz.multivariate.timeseries.UnivariateTimeSeries;
import net.seninp.jmotif.sax.SAXException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.seninp.grammarviz.gi.logic.GIUtils;
import net.seninp.grammarviz.gi.logic.GrammarRuleRecord;
import net.seninp.grammarviz.gi.logic.GrammarRules;
import net.seninp.grammarviz.gi.logic.RuleInterval;
import net.seninp.jmotif.sax.NumerosityReductionStrategy;
import net.seninp.jmotif.sax.SAXProcessor;
import net.seninp.jmotif.sax.alphabet.NormalAlphabet;
import net.seninp.jmotif.sax.datastructure.SAXRecords;


/**
 * Sort of a stand-alone factory to digesting strings with Sequitur.
 *
 * @author psenin
 */
public final class SequiturFactory {

    private static final NormalAlphabet normalA = new NormalAlphabet();

    private static SAXProcessor sp = new SAXProcessor();

    // the logger
    //
    private static final Logger LOGGER = LoggerFactory.getLogger(SequiturFactory.class);

    /**
     * Disabling the constructor.
     */
    private SequiturFactory() {
        assert true;
    }

    /**
     * Digests a string of terminals separated by a space.
     *
     * @param inputString the string to digest.
     * @return The top rule handler (i.e. R0).
     * @throws Exception if error occurs.
     */
    public static SAXRule runSequitur(String inputString) throws Exception {

        LOGGER.trace("digesting the string " + inputString);

        // clear global collections
        //
        SAXRule.numRules = new AtomicInteger(0);
        SAXRule.theRules.clear();
        SAXSymbol.theDigrams.clear();
        SAXSymbol.theSubstituteTable.clear();

        // init the top-level rule
        //
        SAXRule resRule = new SAXRule();

        // tokenize the input string
        //
        StringTokenizer st = new StringTokenizer(inputString, " ");

        // while there are tokens
        int currentPosition = 0;
        while (st.hasMoreTokens()) {

            String token = st.nextToken();
            // System.out.println(" processing the token " + token);

            // extract next token
            SAXTerminal symbol = new SAXTerminal(token, currentPosition);

            // append to the end of the current sequitur string
            // ... As each new input symbol is observed, append it to rule S....
            resRule.last().insertAfter(symbol);

            // once appended, check if the resulting digram is new or recurrent
            //
            // ... Each time a link is made between two symbols if the new digram is repeated elsewhere
            // and the repetitions do not overlap, if the other occurrence is a complete rule,
            // replace the new digram with the non-terminal symbol that heads the rule,
            // otherwise,form a new rule and replace both digrams with the new non-terminal symbol
            // otherwise, insert the digram into the index...
            resRule.last().p.check();

            currentPosition++;

            // LOGGER.debug("Current grammar:\n" + SAXRule.getRules());
        }

        return resRule;
    }



    public static SAXRecords dicretizeTS(UnivariateTimeSeries ts, GrammarVizAnomalyParameters params, NumerosityReductionStrategy strategy) throws SAXException {
        return sp.ts2saxViaWindow(ts.getRawData(), params.SAX_WINDOW_SIZE, params.SAX_PAA_SIZE,
                normalA.getCuts(params.SAX_ALPHABET_SIZE), strategy, params.SAX_NORM_THRESHOLD);
    }


    /**
     * Takes a time series and returns a grammar.
     *
     * @param saxFrequencyData the discretized input time series.
     * @param saxWindowSize    the sliding window size.
     * @param saxPAASize       the PAA num.
     * @return the set of rules, i.e. the grammar.
     */
    public static GrammarRules discretized2SequiturRules(SAXRecords saxFrequencyData, int saxWindowSize,
                                                         int saxPAASize, int tsLength) {
        LOGGER.debug("Inferring the grammar...");
        // this is a string we are about to feed into Sequitur
        //
        String saxDisplayString = saxFrequencyData.getSAXString(" ");

        // reset the Sequitur data structures
        SAXRule.numRules = new AtomicInteger(0);
        SAXRule.theRules.clear();
        SAXSymbol.theDigrams.clear();

        // bootstrap the grammar
        SAXRule grammar = new SAXRule();
        SAXRule.arrRuleRecords = new ArrayList<GrammarRuleRecord>();

        // digest the string via the tokenizer and build the grammar
        StringTokenizer st = new StringTokenizer(saxDisplayString, " ");
        int currentPosition = 0;
        while (st.hasMoreTokens()) {
            grammar.last().insertAfter(new SAXTerminal(st.nextToken(), currentPosition));
            grammar.last().p.check();
            currentPosition++;
        }

        // bw.close();
        LOGGER.debug("Collecting the grammar rules statistics and expanding the rules...");
        GrammarRules rules = grammar.toGrammarRulesData();

        LOGGER.debug("Mapping expanded rules to time-series intervals...");
        SequiturFactory.updateRuleIntervals(rules, saxFrequencyData, true, tsLength, saxWindowSize,
                saxPAASize);

        return rules;

    }

    public static void updateRuleIntervals(GrammarRules rules, SAXRecords saxFrequencyData,
                                           boolean slidingWindowOn, int originalTimeSeriesLength, int saxWindowSize, int saxPAASize) {

        // the original indexes of all SAX words
        ArrayList<Integer> saxWordsIndexes = new ArrayList<Integer>(saxFrequencyData.getAllIndices());

        for (GrammarRuleRecord ruleContainer : rules) {

            // here we construct the array of rule intervals
            ArrayList<RuleInterval> resultIntervals = new ArrayList<RuleInterval>();

            // array of all words of this rule expanded form
            // String[] expandedRuleSplit = ruleContainer.getExpandedRuleString().trim().split(" ");
            int expandedRuleLength = countSpaces(ruleContainer.getExpandedRuleString());

            // the auxiliary array that keeps lengths of all rule occurrences
            int[] lengths = new int[ruleContainer.getOccurrences().size()];
            int lengthCounter = 0;

            // iterate over all occurrences of this rule
            // the currentIndex here is the position of the rule in the input string
            //
            for (Integer currentIndex : ruleContainer.getOccurrences()) {

                // System.out.println("Index: " + currentIndex);
                // String extractedStr = "";

                // what we do here is to extract the positions of sax words in the real time-series
                // by using their positions at the input string
                //
                // int[] extractedPositions = new int[expandedRuleSplit.length];
                // for (int i = 0; i < expandedRuleSplit.length; i++) {
                // extractedStr = extractedStr.concat(" ").concat(
                // saxWordsToIndexesMap.get(saxWordsIndexes.get(currentIndex + i)));
                // extractedPositions[i] = saxWordsIndexes.get(currentIndex + i);
                // }

                int startPos = saxWordsIndexes.get(currentIndex);
                int endPos = -1;
                if ((currentIndex + expandedRuleLength) >= saxWordsIndexes.size()) {
                    endPos = originalTimeSeriesLength;
                } else {
                    if (slidingWindowOn) {
                        endPos = saxWordsIndexes.get(currentIndex + expandedRuleLength) + saxWindowSize - 1;
                    } else {
                        double step = (double) originalTimeSeriesLength / (double) saxPAASize;
                        endPos = Long.valueOf(Math.round(startPos + expandedRuleLength * step)).intValue();
                    }
                }

                resultIntervals.add(new RuleInterval(startPos, endPos));

                lengths[lengthCounter] = endPos - startPos;
                lengthCounter++;
            }
            if (0 == ruleContainer.getRuleNumber()) {
                resultIntervals.add(new RuleInterval(0, originalTimeSeriesLength));
                lengths = new int[1];
                lengths[0] = originalTimeSeriesLength;
            }
            ruleContainer.setRuleIntervals(resultIntervals);
            ruleContainer.setMeanLength((int) GIUtils.mean(lengths));
            ruleContainer.setMinMaxLength(lengths);
        }
    }

    /**
     * Recovers start and stop coordinates of a rule subsequences.
     *
     * @param ruleIdx            The rule index.
     * @param grammar            The grammar to analyze.
     * @param saxFrequencyData   the SAX frquency data used for the grammar construction.
     * @param originalTimeSeries the original time series.
     * @param saxWindowSize      the SAX sliding window size.
     * @return The array of all intervals corresponding to this rule.
     */
    public static ArrayList<RuleInterval> getRulePositionsByRuleNum(int ruleIdx, SAXRule grammar,
                                                                    SAXRecords saxFrequencyData, double[] originalTimeSeries, int saxWindowSize) {

        // this will be the result
        ArrayList<RuleInterval> resultIntervals = new ArrayList<RuleInterval>();

        // the rule container
        GrammarRuleRecord ruleContainer = grammar.getRuleRecords().get(ruleIdx);

        // the original indexes of all SAX words
        ArrayList<Integer> saxWordsIndexes = new ArrayList<Integer>(saxFrequencyData.getAllIndices());

        // debug printout
        LOGGER.trace("Expanded rule: \"" + ruleContainer.getExpandedRuleString() + '\"');
        LOGGER.trace("Indexes: " + ruleContainer.getOccurrences());

        // array of all words of this expanded rule
        String[] expandedRuleSplit = ruleContainer.getExpandedRuleString().trim().split(" ");

        for (Integer currentIndex : ruleContainer.getOccurrences()) {

            String extractedStr = "";
            StringBuffer sb = new StringBuffer(expandedRuleSplit.length);
            for (int i = 0; i < expandedRuleSplit.length; i++) {
                LOGGER.trace("currentIndex " + currentIndex + ", i: " + i);
                extractedStr = extractedStr.concat(" ").concat(String.valueOf(
                        saxFrequencyData.getByIndex(saxWordsIndexes.get(currentIndex + i)).getPayload()));
                sb.append(saxWordsIndexes.get(currentIndex + i)).append(" ");
            }
            LOGGER.trace("Recovered string: " + extractedStr);
            LOGGER.trace("Recovered positions: " + sb.toString());

            int start = saxWordsIndexes.get(currentIndex);
            int end = -1;
            // need to care about bouncing beyond the all SAX words index array
            if ((currentIndex + expandedRuleSplit.length) >= saxWordsIndexes.size()) {
                // if we at the last index - then it's easy - end is the timeseries end
                end = originalTimeSeries.length - 1;
            } else {
                // if we OK with indexes, the Rule subsequence end is the start of the very next SAX word
                // after the kast in this expanded rule
                end = saxWordsIndexes.get(currentIndex + expandedRuleSplit.length) - 1 + saxWindowSize;
            }
            // save it
            resultIntervals.add(new RuleInterval(start, end));
        }

        return resultIntervals;
    }

    /**
     * Counts spaces in the string.
     *
     * @param str The string.
     * @return The number of spaces.
     */
    private static int countSpaces(String str) {
        int counter = 0;
        for (char c : str.toCharArray()) {
            if (c == ' ') {
                counter++;
            }
        }
        return counter;
    }

}
