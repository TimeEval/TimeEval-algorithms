package net.seninp.grammarviz.gi.logic;

import java.util.ArrayList;
import java.util.List;

/**
 * I use this for temporal fixtures.
 * 
 * @author psenin
 * 
 */
public class GIUtils {

  /**
   * Constructor.
   */
  private GIUtils() {
    assert true;
  }

  /**
   * Computes the mean value.
   * 
   * @param values array of values.
   * @return the mean value.
   */
  public static double mean(int[] values) {
    double sum = 0.0;
    for (int i : values) {
      sum = sum + (double) i;
    }
    return sum / (double) values.length;

  }

  /**
   * Run a quick scan along the time series coverage to find a zeroed intervals.
   * 
   * @param coverageArray the coverage to analyze.
   * @return set of zeroed intervals (if found).
   */
  public static List<RuleInterval> getZeroIntervals(int[] coverageArray) {

    ArrayList<RuleInterval> res = new ArrayList<RuleInterval>();

    int start = -1;
    boolean inInterval = false;
    int intervalsCounter = -1;

    // slide over the array from left to the right
    //
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

    // we need to check for the last interval here
    //
    if (inInterval) {
      res.add(new RuleInterval(intervalsCounter, start, coverageArray.length, 0));
    }

    return res;
  }

  /**
   * Computes which fraction of the time series is covered by the rules set.
   * 
   * @param seriesLength the time series length.
   * @param rules the grammar rules set.
   * @return a fraction covered by the rules.
   */
  public static double getCoverAsFraction(int seriesLength, GrammarRules rules) {

    boolean[] coverageArray = new boolean[seriesLength];

    for (GrammarRuleRecord rule : rules) {
      if (0 == rule.ruleNumber()) {
        continue;
      }
      ArrayList<RuleInterval> arrPos = rule.getRuleIntervals();
      for (RuleInterval saxPos : arrPos) {
        int startPos = saxPos.getStart();
        int endPos = saxPos.getEnd();
        for (int j = startPos; j < endPos; j++) {
          coverageArray[j] = true;
        }
      }
    }

    int coverSum = 0;
    for (int i = 0; i < seriesLength; i++) {
      if (coverageArray[i]) {
        coverSum++;
      }
    }
    return (double) coverSum / (double) seriesLength;
  }

  /**
   * Gets the mean rule coverage.
   * 
   * @param length the original time-series length.
   * @param rules the grammar rules set.
   * @return the mean rule coverage.
   */
  public static double getMeanRuleCoverage(int length, GrammarRules rules) {
    // get the coverage array
    //
    int[] coverageArray = new int[length];
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
    // int minCoverage = 0;
    // int maxCoverage = 0;
    int coverageSum = 0;
    for (int i : coverageArray) {
      coverageSum += i;
      // if (i < minCoverage) {
      // minCoverage = i;
      // }
      // if (i > maxCoverage) {
      // maxCoverage = i;
      // }
    }
    return (double) coverageSum / (double) length;
  }

}
