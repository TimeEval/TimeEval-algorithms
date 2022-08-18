package net.seninp.grammarviz.gi.cli;

import java.util.ArrayList;
import net.seninp.grammarviz.gi.logic.GrammarRuleRecord;

/**
 * Computes grammar stats.
 * 
 * @author psenin
 *
 */
public class GrammarStats {

  private static final Object CR = "\n";
  private static final Object TAB = "\t";

  ArrayList<Integer> ruleLength = new ArrayList<Integer>();
  Integer minLength = 0;
  Integer maxLength = 0;

  ArrayList<Integer> ruleUse = new ArrayList<Integer>();
  Integer minUse = 0;
  Integer maxUse = 0;

  ArrayList<Integer> ruleFrequency = new ArrayList<Integer>();
  Integer minFrequency = 0;
  Integer maxFrequency = 0;

  private int ruleCounter = 0;
  private int intervalCounter = 0;

  public void process(GrammarRuleRecord ruleRecord) {

    // don't collect stats for rule #0
    if (0 == ruleRecord.getRuleNumber()) {
      return;
    }

    // LENGTH
    if (this.maxLength < ruleRecord.getMeanLength()) {
      this.maxLength = ruleRecord.getMeanLength();
    }
    if (this.minLength > ruleRecord.getMeanLength()) {
      this.minLength = ruleRecord.getMeanLength();
    }
    this.ruleLength.add(ruleRecord.getMeanLength());

    // USE
    if (this.maxUse < ruleRecord.getRuleUseFrequency()) {
      this.maxUse = ruleRecord.getRuleUseFrequency();
    }
    if (this.minUse > ruleRecord.getRuleUseFrequency()) {
      this.minUse = ruleRecord.getRuleUseFrequency();
    }
    this.ruleUse.add(ruleRecord.getRuleUseFrequency());

    // OCCURRENCE
    int freq = ruleRecord.getRuleIntervals().size();
    if (this.maxFrequency < freq) {
      this.maxFrequency = freq;
    }
    if (this.minFrequency > freq) {
      this.minFrequency = freq;
    }
    this.ruleFrequency.add(freq);

    this.ruleCounter++;
    this.intervalCounter += ruleRecord.getRuleIntervals().size();

  }

  @Override
  public String toString() {

    StringBuilder builder = new StringBuilder();
    builder.append("# GrammarStats:").append(CR);

    builder.append("# rules: ").append(this.ruleCounter).append(CR);
    builder.append("# intervals: ").append(this.intervalCounter).append(CR);

    builder.append("# factor\tmin\tmax\tmean:").append(CR);

    builder.append("# length").append(TAB).append(this.minLength).append(TAB);
    builder.append(this.maxLength).append(TAB).append(mean(this.ruleLength)).append(CR);

    builder.append("# ruleuse").append(TAB).append(this.minUse).append(TAB);
    builder.append(this.maxUse).append(TAB).append(mean(this.ruleUse)).append(CR);

    builder.append("# occurrence").append(TAB).append(this.minFrequency).append(TAB);
    builder.append(this.maxFrequency).append(TAB).append(mean(this.ruleFrequency)).append(CR);

    return builder.toString();
  }

  public String toSingleLine() {

    StringBuilder builder = new StringBuilder();

    builder.append(this.ruleCounter).append(TAB);
    builder.append(this.intervalCounter).append(TAB);

    builder.append(this.minLength).append(TAB);
    builder.append(this.maxLength).append(TAB);
    builder.append(mean(this.ruleLength)).append(TAB);

    builder.append(this.minUse).append(TAB);
    builder.append(this.maxUse).append(TAB);
    builder.append(mean(this.ruleUse)).append(TAB);

    builder.append(this.minFrequency).append(TAB);
    builder.append(this.maxFrequency).append(TAB);
    builder.append(mean(this.ruleFrequency));

    return builder.toString();
  }

  private double mean(ArrayList<Integer> arr) {
    int sum = 0;
    for (int i : arr) {
      sum += i;
    }
    return (double) sum / (double) arr.size();
  }

}
