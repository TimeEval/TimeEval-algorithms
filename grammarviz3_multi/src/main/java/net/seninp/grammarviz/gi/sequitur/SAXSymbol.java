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

import net.seninp.grammarviz.gi.sequitur.SAXGuard;
import net.seninp.grammarviz.gi.sequitur.SAXRule;

import java.util.Hashtable;
import java.util.Map.Entry;

/**
 * Template for Sequitur data structures. Adaption of Eibe Frank code for JMotif API.
 * 
 * @author Manfred Lerner, seninp
 * 
 */
public abstract class SAXSymbol {

  /**
   * Apparently, this limits the possible number of terminals, ids of non-terminals start after this
   * num.
   */
  protected static final int numTerminals = 100000;

  /** Seed the size of hash table? */
  private static final int prime = 2265539;

  /** Hashtable to keep track of all digrams. This is static - single instance for all. */
  protected static final Hashtable<SAXSymbol, SAXSymbol> theDigrams = new Hashtable<SAXSymbol, SAXSymbol>(
      SAXSymbol.prime);

  public static Hashtable<String, Hashtable<String, Integer>> theSubstituteTable = new Hashtable<String, Hashtable<String, Integer>>(
      SAXSymbol.prime);

  /** The symbol value. */
  protected String value;

  /** The symbol original position. */
  protected int originalPosition;

  /** Sort of pointers for previous and the next symbols. */
  public SAXSymbol p;
  public SAXSymbol n;

  /**
   * Links left and right symbols together, i.e. removes this symbol from the string, also removing
   * any old digram from the hash table.
   * 
   * @param left the left symbol.
   * @param right the right symbol.
   */
  public static void join(SAXSymbol left, SAXSymbol right) {

    // System.out.println(" performing the join of " + getPayload(left) + " and "
    // + getPayload(right));

    // check for an OLD digram existence - i.e. left must have a next symbol
    // if .n exists then we are joining TERMINAL symbols within the string, and must clean-up the
    // old digram
    if (left.n != null) {
      // System.out.println(" " + getPayload(left)
      // + " use to be in the digram table, cleaning up");
      left.deleteDigram();
    }

    // re-link left and right
    left.n = right;
    right.p = left;
  }

  /**
   * Cleans up template.
   */
  public abstract void cleanUp();

  /**
   * Inserts a symbol after this one.
   * 
   * @param toInsert the new symbol to be inserted.
   */
  public void insertAfter(SAXSymbol toInsert) {

    // if (this.isGuard()) {
    // System.out.println(" this is Guard of the rule " + ((SAXGuard) this).r.ruleIndex
    // + " inserting " + toInsert.value + " after ");
    // }
    // else if (this.isNonTerminal()) {
    // System.out.println(" this is non-terminal representing the rule "
    // + ((SAXNonTerminal) this).r.ruleIndex + " inserting " + toInsert.value + " after ");
    // }
    // else {
    // System.out.println(" this is symbol " + this.value + " inserting " + toInsert.value
    // + " after ");
    // }

    // call join on this symbol' NEXT - placing it AFTER the new one
    join(toInsert, n);

    // call join on THIS symbol placing the NEW AFTER
    join(this, toInsert);
  }

  /**
   * Removes the digram from the hash table. Overwritten in sub class guard.
   */
  public void deleteDigram() {

    // if N is a Guard - then it is a RULE sits there, don't care about digram
    if (n.isGuard()) {
      return;
    }

    // delete digram if it is exactly this one
    if (this == theDigrams.get(this)) {
      theDigrams.remove(this);
    }
  }

  /**
   * Returns true if this is the guard symbol. Overwritten in subclass guard.
   * 
   * @return true if the guard.
   */
  public boolean isGuard() {
    return false;
  }

  /**
   * Returns true if this is a non-terminal. Overwritten in subclass nonTerminal.
   * 
   * @return true if the non-terminal.
   */
  public boolean isNonTerminal() {
    return false;
  }

  /**
   * "Checks in" a new digram and enforce the digram uniqueness constraint. If it appears elsewhere,
   * deals with it by calling match(), otherwise inserts it into the hash table. Overwritten in
   * subclass guard.
   * 
   * @return true if it is not unique.
   */
  public boolean check() {

    // System.out.println(" performing CHECK on " + getPayload(this));

    // System.out.println("[sequitur debug] *calling check() on* " + this.value + ", n isGuard: "
    // + n.isGuard());

    // ... Each time a link is made between two symbols if the new digram is repeated elsewhere
    // and the repetitions do not overlap, if the other occurrence is a complete rule,
    // replace the new digram with the non-terminal symbol that heads the rule,
    // otherwise,form a new rule and replace both digrams with the new non-terminal symbol
    // otherwise, insert the digram into the index...

    if (n.isGuard()) {
      // i am the rule
      return false;
    }

    if (!theDigrams.containsKey(this)) {
      // System.out.println("[sequitur debug] *check...* digrams contain this (" + this.value + "~"
      // + this.n.value + ")? NO. Checking in.");
      // found = theDigrams.put(this, this);
      theDigrams.put(this, this);
      // System.out.println(" *** Digrams now: " + makeDigramsTable());
      // System.out.println("[sequitur debug] *digrams* " + hash2String());
      return false;
    }

    // System.out.println("[sequitur debug] *check...* digrams contain this (" + this.value
    // + this.n.value + ")? Yes. Oh-Oh...");

    // well the same hash is in the store, lemme see...
    SAXSymbol found = theDigrams.get(this);

    // if it's not me, then lets call match magic?
    if (found.n != this) {
      // System.out.println("[sequitur debug] *double check...* IT IS NOT ME!");
      match(this, found);
    }

    return true;
  }

  // private String hash2String() {
  // StringBuffer sb = new StringBuffer();
  // for (Entry<SAXSymbol, SAXSymbol> e : theDigrams.entrySet()) {
  // // sb.append("[").append(e.getKey().value).append(e.getKey().n.value).append("->");
  // sb.append("[").append(e.getKey().value).append("~").append(e.getKey().n.value).append("->");
  // // sb.append(e.getValue().value).append(e.getKey().n.value).append("],");
  // sb.append(System.identityHashCode(e.getValue())).append("],");
  // }
  // return sb.toString();
  // }

  /**
   * Replace a digram with a non-terminal.
   * 
   * @param r a rule to use.
   */
  public void substitute(SAXRule r) {
    // System.out.println("[sequitur debug] *substitute* " + this.value + " with rule "
    // + r.asDebugLine());
    // clean up this place and the next

    // here we keep the original position in the input string
    //
    r.addIndex(this.originalPosition);

    this.cleanUp();
    this.n.cleanUp();
    // link the rule instead of digram
    SAXNonTerminal nt = new SAXNonTerminal(r);
    nt.originalPosition = this.originalPosition;
    this.p.insertAfter(nt);
    // do p check
    //
    // TODO: not getting this
    if (!p.check()) {
      p.n.check();
    }
  }

  /**
   * Deals with a matching digram.
   * 
   * @param theDigram the first matching digram.
   * @param matchingDigram the second matching digram.
   */
  public void match(SAXSymbol theDigram, SAXSymbol matchingDigram) {

    SAXRule rule;
    SAXSymbol first, second;

    // System.out.println("[sequitur debug] *match* newDigram [" + newDigram.value + ","
    // + newDigram.n.value + "], old matching one [" + matchingDigram.value + ","
    // + matchingDigram.n.value + "]");

    // if previous of matching digram is a guard
    if (matchingDigram.p.isGuard() && matchingDigram.n.n.isGuard()) {
      // reuse an existing rule
      rule = ((SAXGuard) matchingDigram.p).r;
      theDigram.substitute(rule);
    }
    else {
      // well, here we create a new rule because there are two matching digrams
      rule = new SAXRule();

      try {
        // tie the digram's links together within the new rule
        // this uses copies of objects, so they do not get cut out of S
        first = (SAXSymbol) theDigram.clone();
        second = (SAXSymbol) theDigram.n.clone();

        rule.theGuard.n = first;
        first.p = rule.theGuard;
        first.n = second;
        second.p = first;
        second.n = rule.theGuard;
        rule.theGuard.p = second;

        // System.out.println("[sequitur debug] *newRule...* \n" + rule.getRules());

        // put this digram into the hash
        // this effectively erases the OLD MATCHING digram with the new DIGRAM (symbol is wrapped
        // into Guard)
        theDigrams.put(first, first);

        // substitute the matching (old) digram with this rule in S
        // System.out.println("[sequitur debug] *newRule...* substitute OLD digram first.");
        matchingDigram.substitute(rule);

        // substitute the new digram with this rule in S
        // System.out.println("[sequitur debug] *newRule...* substitute NEW digram last.");
        theDigram.substitute(rule);

        // rule.assignLevel();

        // System.out.println(" *** Digrams now: " + makeDigramsTable());

      }
      catch (CloneNotSupportedException c) {
        c.printStackTrace();
      }
    }

    // Check for an underused rule.

    if (rule.first().isNonTerminal() && (((SAXNonTerminal) rule.first()).r.count == 1))
      ((SAXNonTerminal) rule.first()).expand();

    rule.assignLevel();
  }

  /**
   * Custom hashcode implementation. Produces the hashcode for a digram using this and the next
   * symbol.
   * 
   * @return the digram's hash code.
   */
  public int hashCode() {
    int hash1 = 31;
    int hash2 = 13;
    int num0 = 0;

    for (int i = 0; i < value.length(); i++) {
      num0 = num0 + Character.getNumericValue(value.charAt(i));
    }

    int num1 = 0;

    for (int i = 0; i < n.value.length(); i++) {
      num1 = num1 + Character.getNumericValue(n.value.charAt(i));
    }

    hash2 = num0 * hash1 + hash2 * num1;
    return hash2;
  }

  // public int hashCode2() {
  // long code;
  //
  // StringBuilder sb = new StringBuilder(value);
  // sb.append("");
  // sb.append(n.value);
  //
  // code = sb.toString().hashCode();
  // code = code % prime;
  //
  // System.out.println("str:" + sb.toString() + " - str.hc(): " + sb.toString().hashCode()
  // + " - Code: " + code + "(int)Code:" + (int) code);
  //
  // return (int) code;
  // }
  //
  // public int hashCodeOld() {
  //
  // long code;
  //
  // // Values in linear combination with two
  // // prime numbers.
  // // code = ((21599 * (long) value.hashCode()) + (20507 * (long) n.value.hashCode()));
  // code = (long) value.hashCode() + (long) n.value.hashCode();
  // code = code % (long) prime;
  //
  // // System.out.println("value.hc():" + value.hashCode() + " - n.value.hc(): " +
  // // n.value.hashCode()+ " - Code: " + code);
  // return (int) code;
  // }

  /**
   * Test if two digrams are equal. WARNING: don't use to compare two symbols.
   */
  public boolean equals(Object obj) {
    if (this == obj)
      return true;
    if (obj == null)
      return false;
    if (!(obj instanceof SAXSymbol))
      return false;
    // return ((value == ((SAXSymbol)obj).value) &&
    // (n.value == ((SAXSymbol)obj).n.value));
    return ((value.equals(((SAXSymbol) obj).value)) && (n.value.equals(((SAXSymbol) obj).n.value)));
  }

  @Override
  public String toString() {
    // return getPayload(this);
    return "SAXSymbol [value=" + value + ", p=" + p + ", n=" + n + "]";
  }

  /**
   * This routine is used for the debugging.
   * 
   * @param symbol the symbol we looking into.
   * @return symbol's payload.
   */
  protected static String getPayload(SAXSymbol symbol) {
    if (symbol.isGuard()) {
      return "guard of the rule " + ((SAXGuard) symbol).r.ruleIndex;
    }
    else if (symbol.isNonTerminal()) {
      return "nonterminal " + ((SAXNonTerminal) symbol).value;
    }
    return "symbol " + symbol.value;
  }

  @SuppressWarnings("unused")
  private static String makeDigramsTable() {
    StringBuffer sb = new StringBuffer("\n");
    for (Entry<SAXSymbol, SAXSymbol> e : theDigrams.entrySet()) {
      sb.append("           ").append(getPayload(e.getKey())).append(", ")
          .append(getPayload(e.getValue())).append("\n");
    }
    return sb.toString();
  }

}
