package net.seninp.grammarviz.gi.rulepruner;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.seninp.grammarviz.gi.GIAlgorithm;
import net.seninp.grammarviz.gi.logic.GrammarRuleRecord;
import net.seninp.grammarviz.gi.logic.GrammarRules;
import net.seninp.grammarviz.gi.repair.RePairFactory;
import net.seninp.grammarviz.gi.repair.RePairGrammar;
import net.seninp.grammarviz.gi.sequitur.SAXRule;
import net.seninp.grammarviz.gi.sequitur.SequiturFactory;
import net.seninp.jmotif.sax.NumerosityReductionStrategy;
import net.seninp.jmotif.sax.SAXProcessor;
import net.seninp.jmotif.sax.alphabet.NormalAlphabet;
import net.seninp.jmotif.sax.datastructure.SAXRecords;

/**
 * Implements the rule pruner.
 * 
 * @author psenin
 *
 */
public class RulePruner {

  private static final String COMMA = ",";
  private static final String CR = "\n";

  private static final DecimalFormat dfPercent = (new DecimalFormat("0.00"));
  private static final DecimalFormat dfSize = (new DecimalFormat("#.0000"));

  private double[] ts;
  // private SAXProcessor sp;
  // the logger
  //
  private static final Logger LOGGER = LoggerFactory.getLogger(RulePruner.class);
  
    static {
    dfPercent.setDecimalFormatSymbols(new DecimalFormatSymbols(Locale.US));
    dfSize.setDecimalFormatSymbols(new DecimalFormatSymbols(Locale.US));
  }

  public RulePruner(double[] ts) {
    this.ts = ts;
    // this.sp = new SAXProcessor();
  }

  /**
   * Samples the specified point.
   * 
   * @param windowSize the sliding window size.
   * @param paaSize the PAA size.
   * @param alphabetSize the Alphabet size.
   * @param giAlgorithm the GI algorithm to use.
   * @param nrStrategy the numerosity reduction strategy.
   * @param nThreshold the normalization threshold.
   * @return the resulting point characteristics.
   * @throws Exception if error occurs.
   */
  public SampledPoint sample(int windowSize, int paaSize, int alphabetSize, GIAlgorithm giAlgorithm,
      NumerosityReductionStrategy nrStrategy, double nThreshold) throws Exception {
    
    SampledPoint res = new SampledPoint();

    StringBuffer logStr = new StringBuffer();
    logStr.append(windowSize).append(COMMA).append(paaSize).append(COMMA).append(alphabetSize)
        .append(COMMA);

    res.setWindow(windowSize);
    res.setPAA(paaSize);
    res.setAlphabet(alphabetSize);

    // convert to SAX
    //
    // ParallelSAXImplementation ps = new ParallelSAXImplementation();
    SAXProcessor sp = new SAXProcessor();
    NormalAlphabet na = new NormalAlphabet();
    // SAXRecords saxData = ps.process(ts, 1, windowSize, paaSize, alphabetSize, nrStrategy,
    // nThreshold);
    SAXRecords saxData = sp.ts2saxViaWindow(ts, windowSize, paaSize, na.getCuts(alphabetSize),
        nrStrategy, nThreshold);
    if (Thread.currentThread().isInterrupted() && null == saxData) {
      System.err.println("Sampler being interrupted, returning NULL!");
      return null;
    }
    saxData.buildIndex();

    // compute SAX approximation distance
    //
    double approximationDistance = sp.approximationDistancePAA(ts, windowSize, paaSize,
        RulePrunerParameters.SAX_NORM_THRESHOLD)
        + sp.approximationDistanceAlphabet(ts, windowSize, paaSize, alphabetSize,
            RulePrunerParameters.SAX_NORM_THRESHOLD);
    logStr.append(dfSize.format(approximationDistance)).append(COMMA);
    res.setApproxDist(approximationDistance);

    // build a grammar
    //
    GrammarRules rules = new GrammarRules();
    if (GIAlgorithm.SEQUITUR.equals(giAlgorithm)) {
      SAXRule r = SequiturFactory.runSequitur(saxData.getSAXString(" "));
      rules = r.toGrammarRulesData();
      SequiturFactory.updateRuleIntervals(rules, saxData, true, ts.length, windowSize, paaSize);
    }
    else if (GIAlgorithm.REPAIR.equals(giAlgorithm)) {
      RePairGrammar grammar = RePairFactory.buildGrammar(saxData.getSAXString(" "));
      grammar.expandRules();
      grammar.buildIntervals(saxData, ts, windowSize);
      rules = grammar.toGrammarRulesData();
    }

    // compute the grammar size
    //
    Integer grammarSize = RulePrunerFactory.computeGrammarSize(rules, paaSize);
    logStr.append(grammarSize).append(COMMA);
    logStr.append(rules.size()).append(COMMA);
    res.setGrammarSize(grammarSize);
    res.setGrammarRules(rules.size());

    // prune grammar' rules
    //
    GrammarRules prunedRulesSet = RulePrunerFactory.performPruning(ts, rules);
    Integer compressedSize = RulePrunerFactory.computeGrammarSize(prunedRulesSet, paaSize);
    logStr.append(compressedSize).append(COMMA);
    logStr.append(prunedRulesSet.size()).append(COMMA);
    res.setCompressedGrammarSize(compressedSize);
    res.setPrunedRules(prunedRulesSet.size());

    // compute the cover
    //
    boolean[] compressedCover = new boolean[ts.length];
    compressedCover = RulePrunerFactory.updateRanges(compressedCover, prunedRulesSet);
    if (RulePrunerFactory.hasEmptyRanges(compressedCover)) {
      logStr.append("0").append(COMMA);
      res.setCovered(false);
    }
    else {
      logStr.append("1").append(COMMA);
      res.setCovered(true);
    }

    // compute the coverage in percent
    //
    double coverage = RulePrunerFactory.computeCover(compressedCover);
    logStr.append(coverage);
    res.setCoverage(coverage);

    // res.setReduction((double) compressedSize / (double) grammarSize);
    res.setReduction((double) prunedRulesSet.size() / (double) rules.size());

    // get the most frequent rule
    //
    int maxFreq = 0;
    for (GrammarRuleRecord r : prunedRulesSet) {
      if (r.getOccurrences().size() > maxFreq) {
        maxFreq = r.getOccurrences().size();
      }
    }

    res.setMaxFrequency(maxFreq);

    // wrap it up
    //
    logStr.append(CR);

    // print the output
    //
    // bw.write(logStr.toString());
    LOGGER.info(logStr.toString().replace(CR, ""));

    return res;
  }

}
