package net.seninp.tinker;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.seninp.gi.logic.GrammarRuleRecord;
import net.seninp.gi.logic.GrammarRules;
import net.seninp.gi.logic.RuleInterval;
import net.seninp.gi.repair.RePairFactory;
import net.seninp.gi.repair.RePairGrammar;
import net.seninp.gi.rulepruner.RulePrunerFactory;
import net.seninp.grammarviz.GrammarVizAnomaly;
import net.seninp.grammarviz.anomaly.RRAImplementation;
import net.seninp.jmotif.sax.NumerosityReductionStrategy;
import net.seninp.jmotif.sax.TSProcessor;
import net.seninp.jmotif.sax.datastructure.SAXRecords;
import net.seninp.jmotif.sax.discord.DiscordRecord;
import net.seninp.jmotif.sax.discord.DiscordRecords;
import net.seninp.jmotif.sax.parallel.ParallelSAXImplementation;

public class PaperDiscordFinder {

  private static final String IN_DATA = "RCode/TKDD/sine_and_5anomalies.txt";

  private static final String IN_PARAMS = "RCode/TKDD/sine_and_5anomalies_out.txt";

  private static final double normalizationThreshold = 0.05;

  private static final Logger LOGGER = LoggerFactory.getLogger(PaperDiscordFinder.class);

  private static final String COMMA = ",";

  private static final String CR = "\n";

  public static void main(String[] args) throws Exception {

    double[] ts = TSProcessor.readFileColumn(IN_DATA, 0, 0);
    LOGGER.info("read " + ts.length + " points from " + IN_DATA);

    BufferedWriter bw = new BufferedWriter(
        new FileWriter(new File("RCode/TKDD/sine_and_5anomalies_discord_res_10.txt")));

    BufferedReader br = new BufferedReader(new FileReader(new File(IN_PARAMS)));

    String line = null;
    while ((line = br.readLine()) != null) {

      String[] split = line.trim().split(",");
      Integer WIN = Integer.valueOf(split[2]);
      Integer PAA = Integer.valueOf(split[3]);
      Integer ALP = Integer.valueOf(split[4]);

      StringBuilder logStr = new StringBuilder();

      logStr.append(WIN).append(COMMA);
      logStr.append(PAA).append(COMMA);
      logStr.append(ALP).append(COMMA);

      //
      // ************************
      //

      ParallelSAXImplementation ps = new ParallelSAXImplementation();
      SAXRecords parallelRes = ps.process(ts, 2, WIN, PAA, ALP, NumerosityReductionStrategy.NONE,
          0.01);
      RePairGrammar rePairGrammar = RePairFactory.buildGrammar(parallelRes);
      rePairGrammar.expandRules();
      rePairGrammar.buildIntervals(parallelRes, ts, WIN);
      GrammarRules rules = rePairGrammar.toGrammarRulesData();

      // prune grammar' rules
      //
      GrammarRules prunedRulesSet = RulePrunerFactory.performPruning(ts, rules);

      ArrayList<RuleInterval> intervals = new ArrayList<RuleInterval>();

      // populate all intervals with their frequency
      //
      for (GrammarRuleRecord rule : prunedRulesSet) {
        //
        // TODO: do we care about long rules?
        // if (0 == rule.ruleNumber() || rule.getRuleYield() > 2) {
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
      for (GrammarRuleRecord rule : prunedRulesSet) {
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
      List<RuleInterval> zeros = GrammarVizAnomaly.getZeroIntervals(coverageArray);
      if (zeros.size() > 0) {
        intervals.addAll(zeros);
      }

      // run HOTSAX with this intervals set
      //
      DiscordRecords discords = RRAImplementation.series2RRAAnomalies(ts, 10, intervals,
          normalizationThreshold);
      //
      // ************************
      //
      //
      // let's see the discords...

      int[] discordsFound = { 0, 0, 0, 0, 0 };
      ArrayList<Interval> discordIntervals = new ArrayList<Interval>();
      // discord #1
      Interval i = new Interval(205, 217);
      discordIntervals.add(i);
      // discord #2
      i = new Interval(360, 392);
      discordIntervals.add(i);
      // discord #3
      i = new Interval(726, 740);
      discordIntervals.add(i);
      // discord #4
      i = new Interval(500, 545);
      discordIntervals.add(i);
      // discord #5
      i = new Interval(1081, 1095);
      discordIntervals.add(i);

      for (DiscordRecord discord : discords) {

        int pos = discord.getPosition();
        int len = discord.getLength();

        System.out.println(len);

        Interval di = new Interval(pos, pos + len);

        for (int intIdx = 0; intIdx < discordIntervals.size(); intIdx++) {
          Interval interval = discordIntervals.get(intIdx);
          if (interval.intersects(di)) {
            discordsFound[intIdx] = discordsFound[intIdx] + 1;
          }
        }

      }

      for (int j = 0; j < discordsFound.length; j++) {
        logStr.append(discordsFound[j]).append(",");
      }

      logStr.delete(logStr.length() - 1, logStr.length());

      bw.write(logStr.toString() + CR);

    }
    br.close();

    bw.close();
  }

}
