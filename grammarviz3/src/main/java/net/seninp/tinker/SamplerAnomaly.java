package net.seninp.tinker;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.beust.jcommander.JCommander;
import net.seninp.gi.logic.GrammarRuleRecord;
import net.seninp.gi.logic.GrammarRules;
import net.seninp.gi.logic.RuleInterval;
import net.seninp.gi.repair.RePairFactory;
import net.seninp.gi.repair.RePairGrammar;
import net.seninp.grammarviz.anomaly.RRAImplementation;
import net.seninp.jmotif.sax.TSProcessor;
import net.seninp.jmotif.sax.datastructure.SAXRecords;
import net.seninp.jmotif.sax.discord.DiscordRecord;
import net.seninp.jmotif.sax.discord.DiscordRecords;
import net.seninp.jmotif.sax.parallel.ParallelSAXImplementation;
import net.seninp.util.StackTrace;

public class SamplerAnomaly {

  // locale, charset, etc
  //
  final static Charset DEFAULT_CHARSET = StandardCharsets.UTF_8;

  private static final String CR = "\n";

  // static block - we instantiate the logger
  //
  private static final Logger LOGGER = LoggerFactory.getLogger(SamplerAnomaly.class);

  public static void main(String[] args) {

    try {

      SamplerAnomalyParameters params = new SamplerAnomalyParameters();
      JCommander jct = new JCommander(params, args);

      if (0 == args.length) {
        jct.usage();
      }
      else {
        // get params printed

        StringBuffer sb = new StringBuffer(1024);
        sb.append("Sampler anomaly").append(CR);
        sb.append("parameters:").append(CR);
        sb.append("  input file:                  ").append(SamplerAnomalyParameters.IN_FILE)
            .append(CR);
        sb.append("  output file:                 ").append(SamplerAnomalyParameters.OUT_FILE)
            .append(CR);
        sb.append("  SAX sliding window size:     ")
            .append(SamplerAnomalyParameters.SAX_WINDOW_SIZE).append(CR);
        sb.append("  SAX PAA size:                ").append(SamplerAnomalyParameters.SAX_PAA_SIZE)
            .append(CR);
        sb.append("  SAX alphabet size:           ")
            .append(SamplerAnomalyParameters.SAX_ALPHABET_SIZE).append(CR);
        sb.append("  SAX numerosity reduction:    ")
            .append(SamplerAnomalyParameters.SAX_NR_STRATEGY).append(CR);
        sb.append("  SAX normalization threshold: ")
            .append(SamplerAnomalyParameters.SAX_NORM_THRESHOLD).append(CR);

        // read the data
        //
        String dataFName = SamplerAnomalyParameters.IN_FILE;
        double[] ts = TSProcessor.readFileColumn(dataFName, 0, 0);

        // infer the grammar
        //
        ParallelSAXImplementation ps = new ParallelSAXImplementation();
        SAXRecords parallelRes = ps.process(ts, 2, SamplerAnomalyParameters.SAX_WINDOW_SIZE,
            SamplerAnomalyParameters.SAX_PAA_SIZE, SamplerAnomalyParameters.SAX_ALPHABET_SIZE,
            SamplerAnomalyParameters.SAX_NR_STRATEGY, SamplerAnomalyParameters.SAX_NORM_THRESHOLD);
        RePairGrammar rePairGrammar = RePairFactory.buildGrammar(parallelRes);
        rePairGrammar.expandRules();
        rePairGrammar.buildIntervals(parallelRes, ts, SamplerAnomalyParameters.SAX_WINDOW_SIZE);
        GrammarRules rules = rePairGrammar.toGrammarRulesData();

        // GrammarRules rules = SequiturFactory.series2SequiturRules(ts,
        // SamplerAnomalyParameters.SAX_WINDOW_SIZE, SamplerAnomalyParameters.SAX_PAA_SIZE,
        // SamplerAnomalyParameters.SAX_ALPHABET_SIZE, SamplerAnomalyParameters.SAX_NR_STRATEGY,
        // SamplerAnomalyParameters.SAX_NORM_THRESHOLD);

        // populate all intervals with their frequency
        //
        ArrayList<RuleInterval> intervals = new ArrayList<RuleInterval>();
        for (GrammarRuleRecord rule : rules) {
          if (0 == rule.ruleNumber()) {
            continue;
          }
          for (RuleInterval ri : rule.getRuleIntervals()) {
            ri.setCoverage(rule.getRuleIntervals().size());
            ri.setId(rule.ruleNumber());
            intervals.add(ri);
          }
        }

        // get the coverage array
        //
        int[] coverageArray = new int[ts.length];
        for (GrammarRuleRecord rule : rules) {
          if (0 == rule.ruleNumber()) {
            continue;
          }
          ArrayList<RuleInterval> arrPos = rule.getRuleIntervals();
          for (RuleInterval saxPos : arrPos) {
            int startPos = saxPos.getStart();
            int endPos = saxPos.getEnd();
            for (int j = startPos; j < endPos; j++) {
              coverageArray[j] = coverageArray[j] + 1;
            }
          }
        }

        // look for zero-covered intervals and add those to the list
        //
        List<RuleInterval> zeros = getZeroIntervals(coverageArray);
        if (zeros.size() > 0) {
          LOGGER.info("found " + zeros.size() + " intervals not covered by rules: "
              + intervalsToString(zeros));
          intervals.addAll(getZeroIntervals(coverageArray));
        }
        else {
          LOGGER.info("the whole timeseries covered by rule intervals ...");
        }

        // run HOTSAX with this intervals set
        //
        DiscordRecords discords = RRAImplementation.series2RRAAnomalies(ts,
            SamplerAnomalyParameters.DISCORDS_NUM, intervals,
            SamplerAnomalyParameters.SAX_NORM_THRESHOLD);

        // now compose the output file with anomalies
        //
        int[] isAnomaly = new int[ts.length];
        for (int discordId = 0; discordId < discords.getSize(); discordId++) {
          DiscordRecord d = discords.get(discordId);
          for (int i = d.getPosition(); i < d.getPosition() + d.getLength(); i++) {
            isAnomaly[i] = discordId + 1;
          }
        }

        BufferedWriter bw = new BufferedWriter(
            new FileWriter(new File(SamplerAnomalyParameters.OUT_FILE)));
        for (int i : isAnomaly) {
          bw.write(i + "\n");
        }
        bw.close();

      }
    }
    catch (Exception e) {
      System.err.println("error occured while parsing parameters " + Arrays.toString(args) + CR
          + StackTrace.toString(e));
      System.exit(-1);
    }
  }

  /**
   * Run a quick scan along the timeseries coverage to find a zeroed intervals.
   * 
   * @param coverageArray the coverage to analyze.
   * @return set of zeroed intervals (if found).
   */
  private static List<RuleInterval> getZeroIntervals(int[] coverageArray) {
    ArrayList<RuleInterval> res = new ArrayList<RuleInterval>();
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

  /**
   * Makes a zeroed interval to appear nicely in output.
   * 
   * @param zeros the list of zeros.
   * @return the intervals list as a string.
   */
  private static String intervalsToString(List<RuleInterval> zeros) {
    StringBuilder sb = new StringBuilder();
    for (RuleInterval i : zeros) {
      sb.append(i.toString()).append(",");
    }
    return sb.toString();
  }

}
