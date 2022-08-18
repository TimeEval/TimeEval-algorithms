package net.seninp.grammarviz.gi.logic;

import java.io.Serializable;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.SortedMap;
import java.util.TreeMap;

/**
 * Collection of rules, based on the TreeMap to guarantee the iteration order.
 * 
 * @author psenin
 *
 */
public class GrammarRules implements Iterable<GrammarRuleRecord>, Serializable {

  /** The serial. */
  private static final long serialVersionUID = 3982321937773958595L;
  private SortedMap<Integer, GrammarRuleRecord> rules;

  public GrammarRules() {
    super();
    this.rules = new TreeMap<Integer, GrammarRuleRecord>();
  }

  public void addRule(GrammarRuleRecord arrRule) {
    int key = arrRule.getRuleNumber();
    this.rules.put(key, arrRule);
  }

  public GrammarRuleRecord getRuleRecord(Integer ruleIdx) {
    return this.rules.get(ruleIdx);
  }

  public Iterator<GrammarRuleRecord> iterator() {
    return rules.values().iterator();
  }

  public GrammarRuleRecord get(Integer ruleIndex) {
    return rules.get(ruleIndex);
  }

  public int size() {
    return this.rules.size();
  }

  public String toString() {
    StringBuffer sb = new StringBuffer();
    for (Entry<Integer, GrammarRuleRecord> rr : rules.entrySet()) {
      // if(rr.getKey() == 0){ continue; }
      sb.append(rr.getValue().getRuleName());
      sb.append(" -> ").append(rr.getValue().getRuleString());
      sb.append(" -> ").append(rr.getValue().getExpandedRuleString());
      sb.append("\n");
    }
    return sb.delete(sb.length() - 1, sb.length()).toString();
  }

  public int getHighestFrequency() {
    int res = 0;
    for (GrammarRuleRecord r : this.rules.values()) {
      if (0 != r.getRuleNumber()) {
        if (r.getOccurrences().size() > res) {
          res = r.getOccurrences().size();
        }
      }
    }
    return res;
  }
}
