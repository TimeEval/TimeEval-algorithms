package net.seninp.grammarviz.logic;

import net.seninp.gi.logic.RuleInterval;

/**
 * This is a helper data-structure which represents a single occurrence of a rule at the series.
 * 
 * @author Xing Wang, psenin
 * 
 */
public class SAXMotif implements Comparable<SAXMotif> {

  private RuleInterval pos;
  private int ruleIndex;

  private boolean classified = false;
  private SAXMotif similarWith;

  /**
   * @return the pos
   */
  public RuleInterval getPos() {
    return pos;
  }

  /**
   * @param pos the pos to set
   */
  public void setPos(RuleInterval pos) {
    this.pos = pos;
  }

  /**
   * @return the ruleIndex
   */
  public int getRuleIndex() {
    return ruleIndex;
  }

  /**
   * @param ruleIndex the ruleIndex to set
   */
  public void setRuleIndex(int ruleIndex) {
    this.ruleIndex = ruleIndex;
  }

  /**
   * @return the classified
   */
  public boolean isClassified() {
    return classified;
  }

  /**
   * @param classified the classified to set
   */
  public void setClassified(boolean classified) {
    this.classified = classified;
  }

  /**
   * @return the similarWith
   */
  public SAXMotif getSimilarWith() {
    return similarWith;
  }

  /**
   * @param similarWith the similarWith to set
   */
  public void setSimilarWith(SAXMotif similarWith) {
    this.similarWith = similarWith;
  }

  public String toString() {
    return "Rule" + ruleIndex + "\tPosition: " + pos + "\nSimilar With: Rule"
        + similarWith.getRuleIndex() + "\tpos: " + similarWith.getPos() + "\n\n";
  }

  public int compareTo(SAXMotif o) {
    int thisLength = this.pos.getEnd() - this.pos.getStart() + 1;
    int otherLength = o.getPos().getEnd() - o.getPos().getStart() + 1;
    if (thisLength > otherLength) {
      return 1;
    }
    else if (thisLength < otherLength) {
      return -1;
    }
    return -0;
  }
}
