package net.seninp.grammarviz.gi.repair;

import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;
import java.util.StringTokenizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.seninp.jmotif.sax.SAXProcessor;

/**
 * Improved repair implementation.
 * 
 * @author psenin
 *
 */
public class NewRepair {

  private static final String SPACE = " ";

  // the logger
  //
  private static final Logger LOGGER = LoggerFactory.getLogger(NewRepair.class);

  /**
   * Parses the input string into a grammar.
   * 
   * @param inputStr the string to parse.
   * @return the grammar.
   */
  public static RePairGrammar parse(String inputStr) {

    Date start0 = new Date();
    LOGGER.debug("input string (" + String.valueOf(countSpaces(inputStr) + 1) + " tokens) ");

    RePairGrammar grammar = new RePairGrammar();

    // two data structures
    //
    // 1.0. - the string
    ArrayList<RePairSymbolRecord> symbolizedString = new ArrayList<RePairSymbolRecord>(
        countSpaces(inputStr) + 1);

    // 2.0. - the priority queue
    RepairPriorityQueue digramsQueue = new RepairPriorityQueue();

    // 3.0. - the R0 digrams occurrence hashtable: <digram string> -> <R0 occurrence indexes>
    HashMap<String, ArrayList<Integer>> digramsTable = new HashMap<String, ArrayList<Integer>>();

    // 3.1. - all digrams ever seen will be here
    // HashMap<String, ArrayList<Integer>> allDigramsTable = new HashMap<String,
    // ArrayList<Integer>>();

    // tokenize the input string
    StringTokenizer st = new StringTokenizer(inputStr, " ");

    // while there are tokens, populate digrams hash and construct the table
    //
    int stringPositionCounter = 0;
    while (st.hasMoreTokens()) {

      // got token, make a symbol
      String token = st.nextToken();
      RePairSymbol symbol = new RePairSymbol(token, stringPositionCounter);
      // LOGGER.debug("token @" + stringPositionCounter + ": " + token);

      // add it to the string
      RePairSymbolRecord sr = new RePairSymbolRecord(symbol);
      symbolizedString.add(sr);

      // make a digram if we at the second and all consecutive places
      if (stringPositionCounter > 0) {

        // digram str
        StringBuffer digramStr = new StringBuffer();
        digramStr.append(symbolizedString.get(stringPositionCounter - 1).getPayload().toString())
            .append(SPACE)
            .append(symbolizedString.get(stringPositionCounter).getPayload().toString());

        // fill the digram occurrence frequency
        if (digramsTable.containsKey(digramStr.toString())) {
          digramsTable.get(digramStr.toString()).add(stringPositionCounter - 1);
          // LOGGER.debug(" .added a digram entry to: " + digramStr.toString());
        }
        else {
          ArrayList<Integer> arr = new ArrayList<Integer>();
          arr.add(stringPositionCounter - 1);
          digramsTable.put(digramStr.toString(), arr);
          // LOGGER.debug(" .created a digram entry for: " + digramStr.toString());
        }

        symbolizedString.get(stringPositionCounter - 1).setNext(sr);
        sr.setPrevious(symbolizedString.get(stringPositionCounter - 1));

      }

      // go on
      stringPositionCounter++;
    }
    Date start1 = new Date();
    LOGGER.debug("tokenized input and extracted all pairs in "
        + SAXProcessor.timeToString(start0.getTime(), start1.getTime()) + ", " + digramsTable.size()
        + " distinct pairs found");
    // LOGGER.debug("parsed the input string into the doubly linked list of tokens ...");
    // LOGGER.debug("RePair input: " + asString(symbolizedString));
    // LOGGER.debug("digrams table: " + printHash(digramsTable).replace("\n",
    // "\n "));
    // allDigramsTable.putAll(digramsTable);

    // LOGGER.debug("populating the priority queue...");
    // populate the priority queue and the index -> digram record map
    //
    for (Entry<String, ArrayList<Integer>> e : digramsTable.entrySet()) {
      if (e.getValue().size() > 1) {
        // create a digram record
        RepairDigramRecord dr = new RepairDigramRecord(e.getKey(), e.getValue().size());
        // put the record into the priority queue
        digramsQueue.enqueue(dr);
      }
    }
    Date start2 = new Date();
//    LOGGER.debug("built the priority queue in "
//        + SAXProcessor.timeToString(start1.getTime(), start2.getTime()) + ", " + digramsQueue.size()
//        + " digrams in the queue");
    // LOGGER.debug(digramsQueue.toString().replace("\n",
    // "\n "));
    // System.out.println(digramsQueue.toString());

    // start the Re-Pair cycle
    //
    RepairDigramRecord entry = null;
    while ((entry = digramsQueue.dequeue()) != null) {

      // LOGGER.debug(" *the current R0: " + asString(symbolizedString));
      // LOGGER.debug(" *digrams table: " + printHash(digramsTable).replace("\n",
      // "\n "));
      //
      // LOGGER.debug(" *polled a priority queue entry: " + entry.str + " : " + entry.freq);
      // LOGGER.debug(" *" + digramsQueue.toString().replace("\n",
      // "\n "));
      // digramsQueue.runCheck();

      // create a new rule
      //
      ArrayList<Integer> occurrences = digramsTable.get(entry.str);

      RePairSymbolRecord first = symbolizedString.get(occurrences.get(0));
      RePairSymbolRecord second = first.getNext();

      RePairRule r = new RePairRule(grammar);
      // System.out.println("polled a priority queue entry: " + entry.str + " : " + entry.freq + " -> created the rule " + r.ruleNumber);

      r.setFirst(first.getPayload());
      r.setSecond(second.getPayload());
      r.assignLevel();
      r.setExpandedRule(
          first.getPayload().toExpandedString() + SPACE + second.getPayload().toExpandedString());

      // LOGGER.debug(" .creating the rule: " + r.toInfoString());
      //
      // // substitute each digram entry with the rule
      // //
      // LOGGER.debug(" .substituting the digram at locations: " + occurrences.toString());
      HashSet<String> newDigrams = new HashSet<String>(occurrences.size());

      // sometimes we remove some of those...
      ArrayList<Integer> loopOccurrences = new ArrayList<Integer>(occurrences.size());
      for (Integer i : occurrences) {
        loopOccurrences.add(i);
      }
      while (!(loopOccurrences.isEmpty())) {

        // secure the position
        //
        int currentIndex = loopOccurrences.remove(0);
        RePairSymbolRecord currentS = symbolizedString.get(currentIndex);
        RePairSymbolRecord nextS = symbolizedString.get(currentIndex).getNext();

        // 1.0. create a new guard to replace the digram
        //
        RePairGuard g = new RePairGuard(r);
        g.setStringPosition(currentS.getIndex());
        r.addOccurrence(currentS.getIndex());
        RePairSymbolRecord guard = new RePairSymbolRecord(g);
        symbolizedString.set(currentIndex, guard);
        // also place a NULL placeholder next
        RePairSymbolRecord nextNotNull = nextS.getNext();
        guard.setNext(nextNotNull);
        if (null != nextNotNull) {
          nextNotNull.setPrevious(guard);
        }
        RePairSymbolRecord prevNotNull = currentS.getPrevious();
        guard.setPrevious(prevNotNull);
        if (null != prevNotNull) {
          prevNotNull.setNext(guard);
        }

        // 2.0 correct entry at the left
        //
        if (currentIndex > 0 && null != prevNotNull) {

          // cleanup old left digram
          String oldLeftDigram = prevNotNull.getPayload().toString() + " "
              + currentS.getPayload().toString();
          int newFreq = digramsTable.get(oldLeftDigram).size() - 1;
          // consoleLogger
          // .debug(" .removed left digram entry @" + prevNotNull.getPayload().getStringPosition()
          // + " " + oldLeftDigram + ", new freq: " + newFreq);
          digramsTable.get(oldLeftDigram).remove(Integer.valueOf(prevNotNull.getIndex()));
          if (oldLeftDigram.equalsIgnoreCase(entry.str)) {
            loopOccurrences.remove(Integer.valueOf(prevNotNull.getIndex()));
          }
          digramsQueue.updateDigramFrequency(oldLeftDigram, newFreq);

          // if it was the last entry...
          if (0 == newFreq) {
            digramsTable.remove(oldLeftDigram);
            newDigrams.remove(oldLeftDigram);
          }

          // and place the new digram entry
          String newLeftDigram = prevNotNull.getPayload().toString() + " " + r.toString();
          // see the new freq..
          if (digramsTable.containsKey(newLeftDigram)) {
            digramsTable.get(newLeftDigram).add(prevNotNull.getPayload().getStringPosition());
            // LOGGER.debug(" .added a digram entry to: " + newLeftDigram + ", @"
            // + prevNotNull.getPayload().getStringPosition());
          }
          else {
            ArrayList<Integer> arr = new ArrayList<Integer>();
            arr.add(prevNotNull.getPayload().getStringPosition());
            digramsTable.put(newLeftDigram, arr);
            // LOGGER.debug(" .created a digram entry for: " + newLeftDigram.toString()
            // + ", @" + prevNotNull.getPayload().getStringPosition());
          }
          newDigrams.add(newLeftDigram);

        }

        // 3.0 correct entry at the right
        //
        RePairSymbolRecord nextSS = nextS.getNext();
        if (currentIndex < symbolizedString.size() - 2 && null != nextSS) {

          // cleanup old left digram
          String oldRightDigram = nextS.getPayload().toString() + " "
              + nextSS.getPayload().toString();
          int newFreq = digramsTable.get(oldRightDigram).size() - 1;
          // consoleLogger
          // .debug(" .removed right digram entry @" + nextSS.getPayload().getStringPosition()
          // + " " + oldRightDigram + ", new freq: " + newFreq);
          digramsTable.get(oldRightDigram).remove(Integer.valueOf(nextS.getIndex()));
          if (oldRightDigram.equalsIgnoreCase(entry.str)) {
            loopOccurrences.remove(Integer.valueOf(nextS.getIndex()));
          }
          digramsQueue.updateDigramFrequency(oldRightDigram, newFreq);

          // if it was the last entry...
          if (0 == newFreq) {
            digramsTable.remove(oldRightDigram);
            newDigrams.remove(oldRightDigram);
          }

          // and place the new digram entry
          String newRightDigram = r.toString() + " " + nextSS.getPayload().toString();
          // see the new freq..
          if (digramsTable.containsKey(newRightDigram)) {
            digramsTable.get(newRightDigram).add(currentS.getPayload().getStringPosition());
            // LOGGER.debug(" .added a digram entry to: " + newRightDigram + ", @"
            // + currentS.getPayload().getStringPosition());
          }
          else {
            ArrayList<Integer> arr = new ArrayList<Integer>();
            arr.add(currentS.getPayload().getStringPosition());
            digramsTable.put(newRightDigram, arr);
            // LOGGER.debug(" .created a digram entry for: " + newRightDigram.toString()
            // + ", @" + currentS.getPayload().getStringPosition());
          }
          newDigrams.add(newRightDigram);

        }

      } // walk over all occurrences

      // voila -- remove the digram itself from the tracking table
      digramsTable.remove(entry.str);

      // update new digram frequencies and if needed place those into priority queue
      //
      for (String digramStr : newDigrams) {
        if (digramsTable.get(digramStr).size() > 1) {
          if (digramsQueue.containsDigram(digramStr)) {
            digramsQueue.updateDigramFrequency(digramStr, digramsTable.get(digramStr).size());
          }
          else {
            digramsQueue
                .enqueue(new RepairDigramRecord(digramStr, digramsTable.get(digramStr).size()));
          }
        }
      }

    }

    Date start3 = new Date();
    LOGGER.debug("finished repair grammar construction in "
        + SAXProcessor.timeToString(start2.getTime(), start3.getTime()));

    // LOGGER.debug("finished RePair run ...");
    // LOGGER.debug("R0: " + asString(symbolizedString));
    // LOGGER.debug("digrams table: " + printHash(digramsTable).replace("\n",
    // "\n "));
    // LOGGER.debug("digrams queue: " + digramsQueue.toString().replace("\n",
    // "\n "));

    grammar.setR0String(asString(symbolizedString));
    // and since all completed, set the expanded string too
    grammar.setR0ExpnadedString(inputStr.substring(0));

    return grammar;

  }

  // private static String printHash(HashMap<String, ArrayList<Integer>> digramsTable) {
  // StringBuffer sb = new StringBuffer();
  // for (Entry<String, ArrayList<Integer>> e : digramsTable.entrySet()) {
  // sb.append(e.getKey()).append(" -> ").append(e.getValue().toString()).append("\n");
  // }
  // return sb.delete(sb.length() - 1, sb.length()).toString();
  // }

  private static String asString(ArrayList<RePairSymbolRecord> symbolizedString) {
    StringBuffer res = new StringBuffer();
    RePairSymbolRecord s = symbolizedString.get(0); // since digrams are starting from left symbol,
                                                    // the symbol 0 is never NULL
    do {
      res.append(s.getPayload().toString()).append(" ");
      s = s.getNext();
    }
    while (null != s);
    return res.toString();
  }

  /**
   * Counts spaces in the string.
   * 
   * @param str the string to process.
   * @return number of spaces found.
   */
  private static int countSpaces(String str) {
    if (null == str) {
      return -1;
    }
    int counter = 0;
    for (int i = 0; i < str.length(); i++) {
      if (str.charAt(i) == ' ') {
        counter++;
      }
    }
    return counter;
  }
}
