package net.seninp.grammarviz.logic;

/**
 * Container for the rule periodicity data.
 * 
 * @author seninp
 * 
 */
public class RulePeriodicityRecord {

  private int ruleIndex;
  private int ruleFrequency;
  private double length;
  private double period;
  private double periodError;

  public int getRuleIndex() {
    return ruleIndex;
  }

  public void setRuleIndex(int ruleIndex) {
    this.ruleIndex = ruleIndex;
  }

  public int getRuleFrequency() {
    return ruleFrequency;
  }

  public void setRuleFrequency(int ruleFrequency) {
    this.ruleFrequency = ruleFrequency;
  }

  public double getLength() {
    return length;
  }

  public void setRuleLength(double ruleLength) {
    this.length = ruleLength;
  }

  public double getPeriod() {
    return period;
  }

  public void setPeriod(double period) {
    this.period = period;
  }

  public double getPeriodError() {
    return periodError;
  }

  public void setPeriodError(double periodError) {
    this.periodError = periodError;
  }

}
