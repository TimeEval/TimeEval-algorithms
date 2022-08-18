package net.seninp.grammarviz.gi.rulepruner;

import net.seninp.grammarviz.gi.logic.GrammarRuleRecord;
import net.seninp.grammarviz.gi.logic.GrammarRules;
import net.seninp.grammarviz.gi.logic.RuleInterval;
import net.seninp.grammarviz.gi.rulepruner.RulePrunerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Implement algorithm for pruning rules based on optimal covering search.
 */
public class RulePruningAlgorithm {

  private static Logger logger = LoggerFactory.getLogger(RulePruningAlgorithm.class);

  private GrammarRules grammarRules;
  private boolean[] range;
  private Set<Integer> usedRules;
  private Set<Integer> removedRules;

  public RulePruningAlgorithm(GrammarRules grammarRules, int tsLength) {
    this.grammarRules = grammarRules;
    this.range = new boolean[tsLength];

    // these are the rules used in the current cover
    this.usedRules = new HashSet<>();
    usedRules.add(0);

    // these are the rules that were excluded as not contributing anymore
    this.removedRules = new HashSet<>();
  }

  public void pruneRules() {
    // do until all ranges are covered BUT break if no more coverage left
    while (RulePrunerFactory.hasEmptyRanges(range)) {

      GrammarRuleRecord bestRule = this.findRuleWithOptimalCover();
      if (bestRule == null) {
        // i.e. no delta found; no more coverage left
        break;
      }

      usedRules.add(bestRule.getRuleNumber());
      this.removeOverlappingRules();

      // add the new candidate and keep the track of cover
      range = RulePrunerFactory.updateRanges(range, bestRule.getRuleIntervals());
    }

    if (logger.isDebugEnabled()) {
      String rulesAsStrings = Arrays.toString(usedRules.toArray(new Integer[usedRules.size()]));
      logger.debug("Best cover {}", rulesAsStrings);
    }
  }

  /**
   * @return rule with optimal cover, or null if none found (no more coverage left).
   */
  private GrammarRuleRecord findRuleWithOptimalCover() {
    GrammarRuleRecord bestRule = null;
    double bestDelta = Integer.MIN_VALUE;

    for (GrammarRuleRecord rule : grammarRules) {
      int id = rule.getRuleNumber();
      if (!usedRules.contains(id) && !removedRules.contains(id)) {
        double delta = this.getCoverDelta(rule);
        if (delta > bestDelta) {
          bestDelta = delta;
          bestRule = rule;
        }
      }
    }
    return bestRule;
  }

  /**
   * Computes the delta value for the suggested rule candidate.
   *
   * @param rule the grammatical rule candidate.
   * @return the delta value.
   */
  public double getCoverDelta(GrammarRuleRecord rule) {

    // counts which uncovered points shall be covered
    int new_cover = 0;

    // counts overlaps with previously covered ranges
    int overlapping_cover = 0;

    // perform the sum computation
    for (RuleInterval i : rule.getRuleIntervals()) {
      int start = i.getStart();
      int end = i.getEnd();
      for (int j = start; j < end; j++) {
        if (range[j]) {
          overlapping_cover++;
        }
        else {
          new_cover++;
        }
      }
    }

    // if covers nothing, return 0
    if (0 == new_cover) {
      return 0.0;
    }

    // if zero overlap, return full weighted cover
    if (0 == overlapping_cover) {
      return (double) new_cover
              / (double) (rule.getExpandedRuleString().length() + rule.getRuleIntervals().size());
    }

    // else divide newly covered points amount by the sum of the rule string length and occurrence
    // (i.e. encoding size)
    return ((double) new_cover / (double) (new_cover + overlapping_cover))
            / (double) (rule.getExpandedRuleString().length() + rule.getRuleIntervals().size());
  }

  private void removeOverlappingRules() {
    List<RuleInterval> currentCovering = this.usedRuleCovering();
    int[] coveringCounts = this.intervalCoveringCounts(currentCovering);
    int intervalCount = 0;

    boolean continueSearch = true;
    while (continueSearch) {
      continueSearch = false;

      for (int currentRuleId : usedRules) { // used rules are those in the current cover
        if (0 == currentRuleId) {
          continue;
        }

        // the set of intervals in consideration
        GrammarRuleRecord currentRule = grammarRules.get(currentRuleId);
        List<RuleInterval> currentRuleIntervals = currentRule.getRuleIntervals();

        intervalCount -= currentRuleIntervals.size();
        if (intervalCount == 0) {
          break; // this only happens with a single rule, when nothing to compare with
        }
        this.removeFromCoveringCounts(coveringCounts, currentRuleIntervals);

        if (this.isCompletelyCoveredBy(coveringCounts, currentRuleIntervals)) {
          usedRules.remove(currentRuleId);
          removedRules.add(currentRuleId); // we would not consider it later on
          continueSearch = true;
          break;
        }
        else {
          // add the removed rules back into the covering to properly check next rule.
          intervalCount += currentRuleIntervals.size();
          this.updateCoveringCounts(coveringCounts, currentRuleIntervals);
          logger.trace("rule {} can't be removed", currentRule.getRuleName());
        }
      }
    }
  }

  private List<RuleInterval> usedRuleCovering() {
    List<RuleInterval> covering = new ArrayList<>();
    for (int ridB : usedRules) { // used rules are those in the current cover
      if (0 == ridB) {
        continue;
      }
      covering.addAll(grammarRules.get(ridB).getRuleIntervals());
    }
    return covering;
  }

  private int[] intervalCoveringCounts(List<RuleInterval> intervals) {
    int[] coveringCount = new int[range.length];
    this.updateCoveringCounts(coveringCount, intervals);
    return coveringCount;
  }

  private void updateCoveringCounts(int[] covering, List<RuleInterval> intervals) {
    for (RuleInterval i : intervals) {
      for (int j = i.getStart(); j < i.getEnd(); j++) {
        covering[j] += 1;
      }
    }
  }

  private void removeFromCoveringCounts(int[] covering, List<RuleInterval> intervals) {
    for (RuleInterval i : intervals) {
      for (int j = i.getStart(); j < i.getEnd(); j++) {
        covering[j] -= 1;
      }
    }
  }

  /**
   * Checks if the cover is complete.
   *
   * @param isCovered the cover.
   * @param intervals set of rule intervals.
   * @return true if the set complete.
   */
  private boolean isCompletelyCoveredBy(int[] isCovered, List<RuleInterval> intervals) {
    for (RuleInterval i : intervals) {
      for (int j = i.getStart(); j < i.getEnd(); j++) {
          if (isCovered[j] == 0) {
            return false;
          }
      }
    }
    return true;
  }

  public GrammarRules regularizePrunedRules() {
    GrammarRules prunedRules = new GrammarRules();

    for (Integer rId : usedRules) {
      StringBuilder newRuleStr = this.buildExpandedRuleString(rId);
      if (newRuleStr.length() > 0) {
        newRuleStr.delete(newRuleStr.length() - 1, newRuleStr.length());
      }

      GrammarRuleRecord regRule = grammarRules.get(rId);
      regRule.setRuleString(newRuleStr.toString());
      prunedRules.addRule(regRule);
    }

    if (logger.isDebugEnabled()) {
      this.logRegularizationResults();
    }

    return prunedRules;
  }

  private StringBuilder buildExpandedRuleString(Integer rId) {
    String oldRuleStr = grammarRules.get(rId).getRuleString();
    String[] tokens = oldRuleStr.split("\\s+");
    StringBuilder newRuleStr = new StringBuilder();

    for (String t : tokens) {
      if (t.startsWith("R")) {
        Integer ruleId = Integer.valueOf(t.substring(1));
        if (usedRules.contains(ruleId)) {
          newRuleStr.append(t).append(" ");
        } else {
          logger.trace("updating the rule " + rId);
          newRuleStr.append(resolve(ruleId)).append(" ");
        }
      } else {
        newRuleStr.append(t).append(" ");
      }
    }

    return newRuleStr;
  }

  private String resolve(Integer ruleId) {
    StringBuilder newRuleStr = this.buildExpandedRuleString(ruleId);
    if (newRuleStr.length() > 0) {
      newRuleStr.delete(newRuleStr.length() - 1, newRuleStr.length());
    }
    return newRuleStr.toString();
  }

  private void logRegularizationResults() {
    // process the R0 for discrepancies
    //
    // split the rule string onto constituting tokens
    //
    String ruleStr = grammarRules.get(0).getRuleString();
    StringBuilder newRuleString = new StringBuilder();
    String[] tokens = ruleStr.split("\\s+");
    for (String t : tokens) {
      if (t.startsWith("R")) {
        Integer rId = Integer.valueOf(t.substring(1));
        if (usedRules.contains(rId)) {
          newRuleString.append(t).append(" ");
        } else {
          logger.debug("removed rule " + rId + " from R0");
        }
      }
    }

    if (newRuleString.length() > 0) {
      newRuleString.delete(newRuleString.length() - 1, newRuleString.length());
    }

    GrammarRuleRecord newR0 = new GrammarRuleRecord();
    newR0.setRuleNumber(0);
    newR0.setRuleString(newRuleString.toString());
    logger.trace(newR0.toString());
  }
}
