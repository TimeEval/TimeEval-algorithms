package net.seninp.grammarviz.gi.repair;

import java.util.ArrayList;
import net.seninp.grammarviz.gi.logic.RuleInterval;

/**
 * The grammar rule.
 * 
 * @author psenin
 * 
 */
public class RePairRule {

  /** The spacer. */
  private static final char SPACE = ' ';

  /** The global rule enumerator counter. */
  // protected static AtomicInteger numRules = new AtomicInteger(1);

  /** The global rules table. */
  // protected static Hashtable<Integer, RePairRule> theRules = new Hashtable<Integer,
  // RePairRule>();

  /** R0 is important, reserve a var for that. */
  // protected String r0String;
  // protected String r0ExpandedString;

  /** The current rule number. */
  protected int ruleNumber;
  protected String expandedRuleString;

  /** Both symbols, (i.e., pair). */
  protected RePairSymbol first;
  protected RePairSymbol second;
  protected int level;

  /** Occurrences. */
  protected ArrayList<Integer> occurrences;

  /** Which TS interval covered. */
  protected ArrayList<RuleInterval> ruleIntervals;

  /** A handler on the grammar this rule belongs to. */
  private RePairGrammar grammar;

  /**
   * Constructor, assigns a rule ID using the global counter.
   * 
   * @param rg the grammar handler.
   */
  public RePairRule(RePairGrammar rg) {

    this.grammar = rg;

    // assign a next number to this rule and increment the global counter
    this.ruleNumber = rg.numRules.intValue();
    rg.numRules.incrementAndGet();

    rg.theRules.put(this.ruleNumber, this);

    this.occurrences = new ArrayList<Integer>();
    this.ruleIntervals = new ArrayList<RuleInterval>();

  }

  /**
   * First symbol setter.
   * 
   * @param symbol the symbol to set.
   */
  public void setFirst(RePairSymbol symbol) {
    this.first = symbol;
  }

  public RePairSymbol getFirst() {
    return this.first;
  }

  /**
   * Second symbol setter.
   * 
   * @param symbol the symbol to set.
   */
  public void setSecond(RePairSymbol symbol) {
    this.second = symbol;
  }

  public RePairSymbol getSecond() {
    return this.second;
  }

  /**
   * Rule ID getter.
   * 
   * @return the rule ID.
   */
  public int getId() {
    return this.ruleNumber;
  }

  /**
   * Return the prefixed with R rule.
   * 
   * @return rule string.
   */
  public String toRuleString() {
    if (0 == this.ruleNumber) {
      return this.grammar.r0String;
    }
    return this.first.toString() + SPACE + this.second.toString() + SPACE;
  }

  /**
   * Set the expanded rule string.
   * 
   * @param str the expanded rule value.
   * 
   */
  public void setExpandedRule(String str) {
    this.expandedRuleString = str;
  }

  /**
   * Return the prefixed with R rule.
   * 
   * @return rule string.
   */
  public String toExpandedRuleString() {
    return this.expandedRuleString;
  }

  /**
   * Adds a rule occurrence.
   * 
   * @param value the new value.
   */
  public void addOccurrence(int value) {
    if (!this.occurrences.contains(value)) {
      this.occurrences.add(value);
    }
  }

  /**
   * Gets occurrences.
   * 
   * @return all rule's occurrences.
   */
  public int[] getOccurrences() {
    int[] res = new int[this.occurrences.size()];
    for (int i = 0; i < this.occurrences.size(); i++) {
      res[i] = this.occurrences.get(i);
    }
    return res;
  }

  public String toString() {
    return "R" + this.ruleNumber;
  }

  public void assignLevel() {
    int lvl = Integer.MAX_VALUE;
    lvl = Math.min(first.getLevel() + 1, lvl);
    lvl = Math.min(second.getLevel() + 1, lvl);
    this.level = lvl;
  }

  public int getLevel() {
    return this.level;
  }

  public ArrayList<RuleInterval> getRuleIntervals() {
    return this.ruleIntervals;
  }

  public int[] getLengths() {
    if (this.ruleIntervals.isEmpty()) {
      return new int[1];
    }
    int[] res = new int[this.ruleIntervals.size()];
    int count = 0;
    for (RuleInterval ri : this.ruleIntervals) {
      res[count] = ri.getEnd() - ri.getStart();
      count++;
    }
    return res;
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + ((first == null) ? 0 : first.hashCode());
    result = prime * result + ((occurrences == null) ? 0 : occurrences.hashCode());
    result = prime * result + ruleNumber;
    result = prime * result + ((second == null) ? 0 : second.hashCode());
    return result;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj)
      return true;
    if (obj == null)
      return false;
    if (getClass() != obj.getClass())
      return false;
    RePairRule other = (RePairRule) obj;
    if (first == null) {
      if (other.first != null)
        return false;
    }
    else if (!first.equals(other.first))
      return false;
    if (occurrences == null) {
      if (other.occurrences != null)
        return false;
    }
    else if (!occurrences.equals(other.occurrences))
      return false;
    if (ruleNumber != other.ruleNumber)
      return false;
    if (second == null) {
      if (other.second != null)
        return false;
    }
    else if (!second.equals(other.second))
      return false;
    return true;
  }

  public String toInfoString() {
    return this.toString() + " -> " + this.first.toString() + " " + this.second.toString();
  }

}
