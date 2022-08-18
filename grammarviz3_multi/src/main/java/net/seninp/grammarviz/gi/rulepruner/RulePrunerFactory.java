package net.seninp.grammarviz.gi.rulepruner;

import net.seninp.grammarviz.gi.logic.GrammarRuleRecord;
import net.seninp.grammarviz.gi.logic.GrammarRules;
import net.seninp.grammarviz.gi.logic.RuleInterval;

import java.util.Arrays;
import java.util.List;

/**
 * Pruner methods implementation.
 * 
 * @author psenin
 *
 */
public class RulePrunerFactory {

  /**
   * Computes the size of a normal, i.e. unpruned grammar.
   * 
   * @param rules the grammar rules.
   * @param paaSize the SAX transform word size.
   * 
   * @return the grammar size, in BYTES.
   */
  public static Integer computeGrammarSize(GrammarRules rules, Integer paaSize) {

    // The final grammar's size in BYTES
    //
    int res = 0;

    // The final size is the sum of the sizes of all rules
    //
    for (GrammarRuleRecord r : rules) {
      String ruleStr = r.getRuleString();
      String[] tokens = ruleStr.split("\\s+");
      int ruleSize = computeRuleSize(paaSize, tokens);
      res += ruleSize;
    }

    return res;
  }

  private static int computeRuleSize(Integer paaSize, String[] tokens) {
    int ruleSize = 0;
    for (String t : tokens) {
      if (t.startsWith("R")) {
        // if it is a non-terminal, i.e., another rule, we account for a 4 bytes (32 bits
        // offset) pointer onto the rule data structure
        ruleSize = ruleSize + 4;
      }
      else {
        // if it is a terminal, account for a byte used for each letter
        // and 4 bytes for the offset on time-series
        ruleSize = ruleSize + paaSize;
      }
    }
    return ruleSize;
  }

  /**
   * Performs pruning.
   * 
   * @param ts the input time series.
   * @param grammarRules the grammar.
   * @return pruned ruleset.
   */
  public static GrammarRules performPruning(double[] ts, GrammarRules grammarRules) {
    RulePruningAlgorithm pruner = new RulePruningAlgorithm(grammarRules, ts.length);
    pruner.pruneRules();
    return pruner.regularizePrunedRules();
  }

  // /**
  // * Computes the size of a pruned grammar.
  // *
  // * @param ts the input timeseries.
  // * @param rules the grammar rules.
  // * @param paaSize the SAX transform word size.
  // *
  // * @return the grammar size.
  // */
  // public static Integer computePrunedGrammarSize(double[] ts, GrammarRules rules, Integer
  // paaSize) {
  //
  // // res is the final grammar's size
  // //
  // int res = 0;
  //
  // HashSet<Integer> existingRules = new HashSet<Integer>();
  // for (GrammarRuleRecord r : rules) {
  // existingRules.add(r.getRuleNumber());
  // }
  //
  // // first we compute the size needed for encoding of rules
  // //
  // for (GrammarRuleRecord r : rules) {
  //
  // int ruleSize = 0;
  //
  // if (0 == r.getRuleNumber()) {
  // // split the rule string onto constituting tokens
  // //
  // String ruleStr = r.getRuleString();
  // String[] tokens = ruleStr.split("\\s+");
  // for (String t : tokens) {
  // if (t.startsWith("R")) {
  // // it is other rule, so we use a number --> 2 bytes
  // // and pointer on its time-series occurrence
  // ruleSize = ruleSize + 2 + 2;
  // }
  // else {
  // ruleSize = ruleSize + paaSize + 2;
  // }
  // }
  // }
  // else {
  // ruleSize = r.getExpandedRuleString().replaceAll("\\s+", "").length();
  // String ruleStr = r.getRuleString();
  // String[] tokens = ruleStr.split("\\s+");
  // for (String t : tokens) {
  // if (t.startsWith("R") && existingRules.contains(Integer.valueOf(t.substring(1)))) {
  // int expRSize = rules.get(Integer.valueOf(t.substring(1))).getExpandedRuleString()
  // .replaceAll("\\s", "").length();
  // ruleSize = ruleSize - expRSize + 2;
  // }
  // }
  // // ruleSize = ruleSize + r.getOccurrences().size() * 2;
  // }
  //
  // // the increment is computed as the size in bytes which is the sum of:
  // // - the expanded rule string (a letter == byte)
  // // - the number of occurrences * 2 (each occurrence index == a word)
  // // it is safe to skip a space since a word size is fixed
  // //
  // // res = res + r.getExpandedRuleString().replaceAll("\\s", "").length()
  // // + r.getOccurrences().size() * 2;
  // res = res + ruleSize;
  // }
  //
  // // first we compute the cover by rules
  // //
  // // boolean[] range = new boolean[ts.length];
  // // for (GrammarRuleRecord r : rules) {
  // // if (0 == r.getRuleNumber()) {
  // // continue;
  // // }
  // // range = updateRanges(range, r.getRuleIntervals());
  // // }
  //
  // // if happens that not the whole time series is covered, we add the space needed to encode the
  // // gaps
  // // each uncovered point corresponds to a word of length PAA and an index
  // //
  // // if (!(isCovered(range))) {
  // //
  // // for (int i = 0; i < range.length; i++) {
  // // if (false == range[i] && (null != saxData.getByIndex(i))) {
  // // // each uncovered by a rule position is actually an individual PAA word
  // // // and a position index
  // // //
  // // res = res + paaSize + 2;
  // // }
  // // }
  // // }
  //
  // return res;
  // }

  /**
   * Updating the coverage ranges.
   * 
   * @param range the global range array.
   * @param ruleIntervals The intervals used for this update.
   * @return an updated array.
   */
  public static boolean[] updateRanges(boolean[] range, List<RuleInterval> ruleIntervals) {
    boolean[] res = Arrays.copyOf(range, range.length);
    for (RuleInterval i : ruleIntervals) {
      int start = i.getStart();
      int end = i.getEnd();
      for (int j = start; j < end; j++) {
        res[j] = true;
      }
    }
    return res;
  }

  /**
   * Updating the coverage ranges.
   * 
   * @param range the global range array.
   * @param grammar The grammar (i.e. set of rules) used for this update.
   * 
   * @return an updated array.
   */
  public static boolean[] updateRanges(boolean[] range, GrammarRules grammar) {
    boolean[] res = Arrays.copyOf(range, range.length);
    for (GrammarRuleRecord r : grammar) {
      if (0 == r.getRuleNumber()) {
        continue;
      }
      res = updateRanges(res, r.getRuleIntervals());
    }
    return res;
  }

  /**
   * Compute the covered percentage.
   * 
   * @param cover the cover array.
   * @return coverage percentage.
   */
  public static double computeCover(boolean[] cover) {
    int covered = 0;
    for (boolean i : cover) {
      if (i) {
        covered++;
      }
    }
    return (double) covered / (double) cover.length;
  }

  /**
   * Checks if the range is completely covered.
   * 
   * @param range the range.
   * @return true if covered.
   */
  public static boolean isCovered(boolean[] range) {
    for (boolean i : range) {
      if (!i) {
        return false;
      }
    }
    return true;
  }

  /**
   * Searches for empty (i.e. uncovered) ranges.
   * 
   * @param range the whole range to analyze.
   * 
   * @return true if uncovered ranges exist.
   */
  public static boolean hasEmptyRanges(boolean[] range) {
    //
    // the visual debugging
    //
    // StringBuffer sb = new StringBuffer();
    // boolean inUncovered = false;
    // int start = 0;
    // for (int i = 0; i < range.length; i++) {
    // if (false == range[i] && false == inUncovered) {
    // start = i;
    // inUncovered = true;
    // }
    // if (true == range[i] && true == inUncovered) {
    // sb.append("[" + start + ", " + i + "], ");
    // inUncovered = false;
    // }
    // }
    // if (inUncovered) {
    // sb.append("[" + start + ", " + range.length + "], ");
    // }
    // System.out.println(sb);
    //
    //
    for (boolean p : range) {
      if (!p) {
        return true;
      }
    }
    return false;
  }
}
