package net.seninp.grammarviz.gi.sequitur;

/*
 This class is part of a Java port of Craig Nevill-Manning's Sequitur algorithm.
 Copyright (C) 1997 Eibe Frank

 This program is free software; you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation; either version 2
 of the License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Set;
import java.util.TreeSet;
import java.util.Vector;
import java.util.concurrent.atomic.AtomicInteger;
import net.seninp.grammarviz.gi.logic.GrammarRuleRecord;
import net.seninp.grammarviz.gi.logic.GrammarRules;

/**
 * The Rule. Adaption of Eibe Frank code for JMotif API.
 * 
 * @author Manfred Lerner, seninp
 * 
 */
public class SAXRule {

  //
  // The rule utility constraint demands that a rule be deleted if it is referred to only once. Each
  // rule has an associated reference count, which is incremented when a non-terminal symbol that
  // references the rule is created, and decremented when the non-terminal symbol is deleted. When
  // the reference count falls to one, the rule is deleted.

  /** This is static - the global rule enumerator counter. */
  protected static AtomicInteger numRules = new AtomicInteger(0);

  /** Yet another, global static structure allowing fast rule access. */
  protected static final ArrayList<SAXRule> theRules = new ArrayList<SAXRule>();

  private static final String SPACE = " ";

  private static final String TAB = "\t";

  private static final String CR = "\n";

  /** Keeper for rules references. */
  protected static ArrayList<GrammarRuleRecord> arrRuleRecords = new ArrayList<GrammarRuleRecord>();

  /** Guard symbol to mark beginning and end of rule. */
  protected SAXGuard theGuard;

  /** Counter keeps track of how many times the rule is used in the grammar. */
  protected int count;

  /** The rule's number. Used for identification of non-terminals. */
  protected int ruleIndex;

  /** Index used for printing. */
  protected int index;

  /** The rule level, used to assess the hierarchy. */
  protected int level;

  /**
   * This keeps rule indexes - once rule created or used, its placement position is extracted from
   * the TerminalSymbol position and stored here.
   */
  protected Set<Integer> indexes = new TreeSet<Integer>();

  /**
   * Constructor.
   */
  public SAXRule() {

    // assign a next number to this rule and increment the global counter
    this.ruleIndex = numRules.intValue();
    numRules.incrementAndGet();

    // create a Guard handler for the rule
    this.theGuard = new SAXGuard(this);

    // init other vars
    this.count = 0;
    this.index = 0;
    this.level = 0;

    // save the instance
    theRules.add(this);
  }

  /**
   * Original getRules() method. Prints out rules. Killing it will brake tests.
   *
   * @return the formatted rules string.
   */
  public static String printRules() {

    theRules.get(0).getSAXRules();
    expandRules();

    Vector<SAXRule> rules = new Vector<SAXRule>(numRules.intValue());
    SAXRule currentRule;
    SAXRule referedTo;
    SAXSymbol sym;
    int index;
    int processedRules = 0;
    StringBuilder text = new StringBuilder();

    text.append("Number\tName\tLevel\tOccurr.\tUsage\tYield\tRule str\tExpaneded\tIndexes\n");
    rules.addElement(theRules.get(0));

    // add-on - keeping the rule string, will be used in order to expand rules
    StringBuilder currentRuleString = new StringBuilder();

    while (processedRules < rules.size()) {

      currentRule = rules.elementAt(processedRules);

      // seninp: adding to original output rule occurrence indexes
      //
      text.append(SPACE);
      text.append(arrRuleRecords.get(processedRules).getRuleNumber()).append(TAB);
      text.append(arrRuleRecords.get(processedRules).getRuleName()).append(TAB);
      text.append(arrRuleRecords.get(processedRules).getRuleLevel()).append(TAB);
      text.append(arrRuleRecords.get(processedRules).getOccurrences().size()).append(TAB);
      text.append(arrRuleRecords.get(processedRules).getRuleUseFrequency()).append(TAB);
      text.append(arrRuleRecords.get(processedRules).getRuleYield()).append(TAB);

      for (sym = currentRule.first(); (!sym.isGuard()); sym = sym.n) {
        if (sym.isNonTerminal()) {
          referedTo = ((SAXNonTerminal) sym).r;
          if ((rules.size() > referedTo.index) && (rules.elementAt(referedTo.index) == referedTo)) {
            index = referedTo.index;
          }
          else {
            index = rules.size();
            referedTo.index = index;
            rules.addElement(referedTo);
          }
          text.append('R');
          text.append(index);

          currentRuleString.append('R');
          currentRuleString.append(index);
        }
        else {
          if (sym.value.equals(" ")) {
            text.append('_');
            currentRuleString.append('_');
          }
          else {
            if (sym.value.equals("\n")) {
              text.append("\\n");
              currentRuleString.append("\\n");
            }
            else {
              text.append(sym.value);
              currentRuleString.append(sym.value);
            }
          }
        }
        text.append(' ');
        currentRuleString.append(' ');
      }
      text.append(TAB).append(arrRuleRecords.get(processedRules).getExpandedRuleString())
          .append(TAB);
      text.append(Arrays.toString(currentRule.getIndexes())).append(CR);

      processedRules++;

      currentRuleString = new StringBuilder();
    }
    return text.toString();
  }

  /**
   * Cleans up data structures.
   */
  public static void reset() {
    SAXRule.numRules = new AtomicInteger(0);
    SAXSymbol.theDigrams.clear();
    SAXSymbol.theSubstituteTable.clear();
    SAXRule.arrRuleRecords = new ArrayList<GrammarRuleRecord>();
  }

  /**
   * Report the FIRST symbol of the rule.
   * 
   * @return the FIRST rule's symbol.
   */
  public SAXSymbol first() {
    return this.theGuard.n;
  }

  /**
   * Report the LAST symbol of the rule.
   * 
   * @return the LAST rule's symbol.
   */
  public SAXSymbol last() {
    return this.theGuard.p;
  }

  // /**
  // * this function iterates over the SAX containers with the Sequitur rules rules of a time series
  // * and connects the single SAX strings of a rule so that the rule can be identified. Therefore a
  // * connect string or an empty string can be used
  // *
  // * Example: Sequitur rule: aabb ccaa bbcc concatentionString: "#" rule after concatenation:
  // * aabb#ccaa#bbcc
  // *
  // * this is done for all rules of a time series
  // *
  // * @param concatenationString the string which connects the single words of a Sequitur rule
  // * @return the SAXified time series with all the rules in it as expanded string
  // */
  // public String getSequiturDocument(String concatenationString) {
  // HashMap<String, String> ruleMap = new HashMap<String, String>();
  //
  // for (SAXRuleRecord cont : arrSAXRuleRecords) {
  // ruleMap.put(cont.getRuleName(), cont.getExpandedRuleString());
  // }
  //
  // String rule0 = arrSAXRuleRecords.get(0).getRuleString();
  // StringTokenizer st = new StringTokenizer(rule0, " ");
  //
  // StringBuilder sbDocument = new StringBuilder();
  // while (st.hasMoreTokens()) {
  // String token = st.nextToken();
  //
  // if (ruleMap.containsKey(token) == false) {
  // sbDocument.append(token);
  // sbDocument.append(" ");
  // continue;
  // }
  //
  // String rule = ruleMap.get(token);
  // rule = rule.replace(" ", concatenationString);
  //
  // // delete final concatenation character
  // if (rule.endsWith(concatenationString)) {
  // rule = rule.substring(0, rule.length() - 1);
  // }
  // sbDocument.append(rule);
  // sbDocument.append(" ");
  // }
  //
  // return sbDocument.toString();
  // }

  /**
   * This traces the rule level.
   */
  protected void assignLevel() {

    int lvl = Integer.MAX_VALUE;

    SAXSymbol sym;

    for (sym = this.first(); (!sym.isGuard()); sym = sym.n) {

      if (sym.isNonTerminal()) {
        SAXRule referedTo = ((SAXNonTerminal) sym).r;
        lvl = Math.min(referedTo.level + 1, lvl);
      }
      else {
        level = 1;
        return;
      }

    }

    level = lvl;
  }

  public int getLevel() {
    return this.level;
  }

  /**
   * Manfred's cool trick to get out all expanded rules. Expands the rule of each SAX container into
   * SAX words string. Can be rewritten recursively though.
   */
  private static void expandRules() {

    // long start = System.currentTimeMillis();

    // iterate over all SAX containers
    // ArrayList<SAXMapEntry<Integer, Integer>> recs = new ArrayList<SAXMapEntry<Integer, Integer>>(
    // arrRuleRecords.size());
    //
    // for (GrammarRuleRecord ruleRecord : arrRuleRecords) {
    // recs.add(new SAXMapEntry<Integer, Integer>(ruleRecord.getRuleLevel(), ruleRecord
    // .getRuleNumber()));
    // }
    //
    // Collections.sort(recs, new Comparator<SAXMapEntry<Integer, Integer>>() {
    // @Override
    // public int compare(SAXMapEntry<Integer, Integer> o1, SAXMapEntry<Integer, Integer> o2) {
    // return o1.getKey().compareTo(o2.getKey());
    // }
    // });

    // for (SAXMapEntry<Integer, Integer> entry : recs) {
    for (GrammarRuleRecord ruleRecord : arrRuleRecords) {

      if (ruleRecord.getRuleNumber() == 0) {
        continue;
      }

      String curString = ruleRecord.getRuleString();
      StringBuilder resultString = new StringBuilder(8192);

      String[] split = curString.split(" ");

      for (String s : split) {
        if (s.startsWith("R")) {
          resultString.append(" ").append(expandRule(Integer.valueOf(s.substring(1, s.length()))));
        }
        else {
          resultString.append(" ").append(s);
        }
      }

      // need to trim space at the very end
      String rr = resultString.delete(0, 1).append(" ").toString();
      ruleRecord.setExpandedRuleString(rr);
      ruleRecord.setRuleYield(countSpaces(rr));
    }

    StringBuilder resultString = new StringBuilder(8192);

    GrammarRuleRecord ruleRecord = arrRuleRecords.get(0);
    resultString.append(ruleRecord.getRuleString());

    int currentSearchStart = resultString.indexOf("R");
    while (currentSearchStart >= 0) {
      int spaceIdx = resultString.indexOf(" ", currentSearchStart);
      String ruleName = resultString.substring(currentSearchStart, spaceIdx + 1);
      Integer ruleId = Integer.valueOf(ruleName.substring(1, ruleName.length() - 1));
      resultString.replace(spaceIdx - ruleName.length() + 1, spaceIdx + 1,
          arrRuleRecords.get(ruleId).getExpandedRuleString());
      currentSearchStart = resultString.indexOf("R");
    }
    ruleRecord.setExpandedRuleString(resultString.toString().trim());
    // ruleRecord.setRuleYield(countSpaces(resultString));

    // long end = System.currentTimeMillis();
    // System.out.println("Rules expanded in " + SAXFactory.timeToString(start, end));

  }

  private static String expandRule(Integer ruleNum) {
    GrammarRuleRecord rr = arrRuleRecords.get(ruleNum);

    String curString = rr.getRuleString();
    StringBuilder resultString = new StringBuilder();

    String[] split = curString.split(" ");

    for (String s : split) {
      if (s.startsWith("R")) {
        resultString.append(" ").append(expandRule(Integer.valueOf(s.substring(1, s.length()))));
      }
      else {
        resultString.append(" ").append(s);
      }
    }
    String res = resultString.delete(0, 1).append(" ").toString();
    rr.setExpandedRuleString(res);
    return resultString.delete(resultString.length() - 1, resultString.length()).toString();
  }

  /**
   * Counts spaces in the string.
   * 
   * @param str The string.
   * @return The number of spaces.
   */
  private static int countSpaces(String str) {
    int counter = 0;
    for (int i = 0; i < str.length(); i++) {
      if (str.charAt(i) == ' ') {
        counter++;
      }
    }
    return counter;
  }

  public ArrayList<GrammarRuleRecord> getRuleRecords() {
    return arrRuleRecords;
  }

  /**
   * Adds an index of the rule occurrence.
   * 
   * @param position the rule position.
   */
  public void addIndex(int position) {
    // save the index at the input string
    //
    this.indexes.add(position);
  }

  /**
   * Get all the rule occurrences.
   * 
   * @return all the rule occurrences.
   */
  private int[] getIndexes() {
    int[] res = new int[this.indexes.size()];
    int i = 0;
    for (Integer idx : this.indexes) {
      res[i] = idx;
      i++;
    }
    return res;
  }

  /**
   * Add-on to the original code by manfred and seninp. This one similar to the original getRules()
   * but populates and returns the array list of SAXRuleRecords.
   */
  protected void getSAXRules() {

    arrRuleRecords.clear();

    Vector<SAXRule> rules = new Vector<SAXRule>(numRules.intValue());
    rules.addElement(this);

    SAXRule currentRule;

    int processedRules = 0;

    StringBuilder sbCurrentRule = new StringBuilder();

    while (processedRules < rules.size()) {

      currentRule = rules.elementAt(processedRules);

      for (SAXSymbol sym = currentRule.first(); (!sym.isGuard()); sym = sym.n) {
        if (sym.isNonTerminal()) {
          SAXRule referedTo = ((SAXNonTerminal) sym).r;
          if ((rules.size() > referedTo.index) && (rules.elementAt(referedTo.index) == referedTo)) {
            index = referedTo.index;
          }
          else {
            index = rules.size();
            referedTo.index = index;
            rules.addElement(referedTo);
          }
          sbCurrentRule.append('R');
          sbCurrentRule.append(index);
        }
        else {
          sbCurrentRule.append(sym.value);
        }
        sbCurrentRule.append(' ');
      }

      GrammarRuleRecord ruleConteiner = new GrammarRuleRecord();

      ruleConteiner.setRuleNumber(processedRules);
      ruleConteiner.setRuleString(sbCurrentRule.toString());

      ruleConteiner.setRuleLevel(currentRule.getLevel());

      ruleConteiner.setRuleUseFrequency(currentRule.count);
      ruleConteiner.setOccurrences(currentRule.getIndexes());

      arrRuleRecords.add(ruleConteiner);

      sbCurrentRule = new StringBuilder();
      processedRules++;
    }

  }

  public GrammarRules toGrammarRulesData() {
    getSAXRules();
    expandRules();
    GrammarRules res = new GrammarRules();
    for (GrammarRuleRecord arrRule : arrRuleRecords) {
      res.addRule(arrRule);
    }
    return res;
  }

  // /**
  // * Original getRules() method. Prints out rules. Killing it will brake tests.
  // *
  // * @return the formatted rules string.
  // */
  // public static String printRules() {
  //
  // theRules.get(0).getSAXRules();
  // expandRules();
  //
  // Vector<SAXRule> rules = new Vector<SAXRule>(numRules.intValue());
  // SAXRule currentRule;
  // SAXRule referedTo;
  // SAXSymbol sym;
  // int index;
  // int processedRules = 0;
  // StringBuilder text = new StringBuilder();
  //
  // text.append("Number\tName\tLevel\tOccurr.\tUsage\tYield\tRule str\tExpaneded\tIndexes\n");
  // rules.addElement(theRules.get(0));
  //
  // // add-on - keeping the rule string, will be used in order to expand rules
  // StringBuilder currentRuleString = new StringBuilder();
  //
  // while (processedRules < rules.size()) {
  //
  // currentRule = rules.elementAt(processedRules);
  //
  // // seninp: adding to original output rule occurrence indexes
  // //
  // text.append(SPACE);
  // text.append(arrRuleRecords.get(processedRules).getRuleNumber()).append(TAB);
  // text.append(arrRuleRecords.get(processedRules).getRuleName()).append(TAB);
  // text.append(arrRuleRecords.get(processedRules).getRuleLevel()).append(TAB);
  // text.append(arrRuleRecords.get(processedRules).getOccurrences().size()).append(TAB);
  // text.append(arrRuleRecords.get(processedRules).getRuleUseFrequency()).append(TAB);
  // text.append(arrRuleRecords.get(processedRules).getRuleYield()).append(TAB);
  //
  // for (sym = currentRule.first(); (!sym.isGuard()); sym = sym.n) {
  // if (sym.isNonTerminal()) {
  // referedTo = ((SAXNonTerminal) sym).r;
  // if ((rules.size() > referedTo.index) && (rules.elementAt(referedTo.index) == referedTo)) {
  // index = referedTo.index;
  // }
  // else {
  // index = rules.size();
  // referedTo.index = index;
  // rules.addElement(referedTo);
  // }
  // text.append('R');
  // text.append(index);
  //
  // currentRuleString.append('R');
  // currentRuleString.append(index);
  // }
  // else {
  // if (sym.value.equals(" ")) {
  // text.append('_');
  // currentRuleString.append('_');
  // }
  // else {
  // if (sym.value.equals("\n")) {
  // text.append("\\n");
  // currentRuleString.append("\\n");
  // }
  // else {
  // text.append(sym.value);
  // currentRuleString.append(sym.value);
  // }
  // }
  // }
  // text.append(' ');
  // currentRuleString.append(' ');
  // }
  // text.append(TAB).append(arrRuleRecords.get(processedRules).getExpandedRuleString())
  // .append(TAB);
  // text.append(Arrays.toString(currentRule.getIndexes())).append(CR);
  //
  // processedRules++;
  //
  // currentRuleString = new StringBuilder();
  // }
  // return text.toString();
  // }

}
