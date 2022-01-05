package net.seninp.grammarviz.anomaly;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.HashSet;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.seninp.gi.logic.RuleInterval;
import net.seninp.jmotif.distance.EuclideanDistance;
import net.seninp.jmotif.sax.SAXProcessor;
import net.seninp.jmotif.sax.TSProcessor;
import net.seninp.jmotif.sax.discord.DiscordRecord;
import net.seninp.jmotif.sax.discord.DiscordRecords;
import net.seninp.grammarviz.GrammarVizAnomaly;

/**
 * Implements RRA algorithm.
 *
 * @author psenin
 *
 */
public class RRAImplementation {

  private static TSProcessor tp = new TSProcessor();
  private static EuclideanDistance ed = new EuclideanDistance();

  // static block - we instantiate the logger
  //
  private static final Logger LOGGER = LoggerFactory.getLogger(RRAImplementation.class);

  /**
   * Implements RRA -- an anomaly discovery algorithm based on discretization and grammar inference.
   * RRA stands for rare rule anomaly.
   *
   * @param series The series to find discord at.
   * @param discordCollectionSize How many discords to find.
   * @param intervals The intervals. In our implementation these come from the set of Sequitur
   * grammar rules.
   * @param zNormThreshold - the normalization threshold (dstance).
   * @return Discords.
   * @throws TSException If error occurs.
   */
  public static DiscordRecords series2RRAAnomalies(double[] series, int discordCollectionSize,
      ArrayList<RuleInterval> intervals, double zNormThreshold) throws Exception {

    Date gStart = new Date();

    // resulting discords collection
    DiscordRecords discords = new DiscordRecords();

    if (intervals.isEmpty()) {
      return discords;
    }

    // visit registry
    HashSet<Integer> registry = new HashSet<Integer>(
        discordCollectionSize * intervals.get(0).getLength() * 2);

    // we conduct the search until the number of discords is less than desired
    //
    while (discords.getSize() < discordCollectionSize) {

      LOGGER.trace(
          "currently known discords: " + discords.getSize() + " out of " + discordCollectionSize);

      Date start = new Date();
      DiscordRecord bestDiscord = findBestDiscordForIntervals(series, intervals, registry,
          zNormThreshold);
      Date end = new Date();

      // if the discord is null we getting out of the search
      if (bestDiscord.getNNDistance() == Integer.MIN_VALUE
          || bestDiscord.getPosition() == Integer.MIN_VALUE) {
        LOGGER.trace("breaking the outer search loop, discords found: " + discords.getSize()
            + " last seen discord: " + bestDiscord.toString());
        break;
      }

      bestDiscord.setInfo("position " + bestDiscord.getPosition() + ", length "
          + bestDiscord.getLength() + ", NN distance " + bestDiscord.getNNDistance()
          + ", elapsed time: " + SAXProcessor.timeToString(start.getTime(), end.getTime()) + ", "
          + bestDiscord.getInfo());
      LOGGER.debug(bestDiscord.getInfo());

      // collect the result
      //
      discords.add(bestDiscord);

      // mark the discord discovered
      //
      int markStart = bestDiscord.getPosition() - bestDiscord.getLength();
      int markEnd = bestDiscord.getPosition() + bestDiscord.getLength();
      if (markStart < 0) {
        markStart = 0;
      }
      if (markEnd > series.length) {
        markEnd = series.length;
      }
      for (int i = markStart; i < markEnd; i++) {
        registry.add(i);
      }
    }

    LOGGER.info(discords.getSize() + " discords found in "
        + SAXProcessor.timeToString(gStart.getTime(), new Date().getTime()));

    // done deal
    //
    return discords;
  }

  /**
   *
   * @param series
   * @param globalIntervals
   * @param registry
   * @param zNormThreshold
   * @return
   * @throws Exception
   */
  public static DiscordRecord findBestDiscordForIntervals(double[] series,
      ArrayList<RuleInterval> globalIntervals, HashSet<Integer> registry, double zNormThreshold)
      throws Exception {

    // prepare the visits array, note that there can't be more points to visit that in a SAX index
    int[] visitArray = new int[globalIntervals.size()];

    // this is outer loop heuristics
    ArrayList<RuleInterval> intervals = cloneIntervals(globalIntervals);
    Collections.sort(intervals, new Comparator<RuleInterval>() {
      public int compare(RuleInterval c1, RuleInterval c2) {
        return Double.compare(c1.getCoverage(), c2.getCoverage());
      }
    });

    // init variables
    int bestSoFarPosition = Integer.MIN_VALUE;
    int bestSoFarLength = Integer.MIN_VALUE;
    int bestSoFarRule = Integer.MIN_VALUE;

    double bestSoFarDistance = Integer.MIN_VALUE;

    // we will iterate over words from rarest to frequent ones - this is an OUTER LOOP of the best
    // discord search
    //
    int iterationCounter = 0;
    int distanceCalls = 0;

    LOGGER
        .trace("going to iterate over " + intervals.size() + " intervals looking for the discord");

    for (int i = 0; i < intervals.size(); i++) {

      iterationCounter++;

      RuleInterval currentEntry = intervals.get(i);

      // make sure it is not a previously found discord
      if (registry.contains(currentEntry.getStart())) {
        continue;
      }

      int currentPos = currentEntry.getStart();
      String currentRule = String.valueOf(currentEntry.getId());

      LOGGER.trace("iteration " + i + ", out of " + intervals.size() + ", rule " + currentRule
          + " at " + currentPos + ", length " + currentEntry.getLength());

      // other occurrences of the current rule
      // TODO : this can be taken out of here to optimize multiple discords discovery
      ArrayList<Integer> currentOccurences = listRuleOccurrences(currentEntry.getId(), intervals);
      LOGGER.trace(" there are " + currentOccurences.size() + " occurrences for the rule "
          + currentEntry.getId() + ", iterating...");

      // organize visited so-far positions tracking
      //
      int markStart = currentPos - currentEntry.getLength();
      if (markStart < 0) {
        markStart = 0;
      }
      int markEnd = currentPos + currentEntry.getLength();
      if (markEnd > series.length) {
        markEnd = series.length;
      }

      // all the candidates we are not going to try
      HashSet<Integer> alreadyVisited = new HashSet<Integer>(
          currentOccurences.size() + (markEnd - markStart));
      for (int j = markStart; j < markEnd; j++) {
        alreadyVisited.add(j);
      }

      // extract the subsequence & mark visited current substring
      // double[] currentSubsequence = tp.subseriesByCopy(series, currentEntry.getStart(),
      // currentEntry.getEnd());

      // so, lets the search begin...
      double nearestNeighborDist = Double.MAX_VALUE;
      boolean doRandomSearch = true;

      // this is the first INNER LOOP
      for (Integer nextOccurrenceIdx : currentOccurences) {

        RuleInterval nextOccurrence = intervals.get(nextOccurrenceIdx);

        // skip the location we standing at, check if we overlap
        if (alreadyVisited.contains(nextOccurrence.getStart())) {
          continue;
        }
        else {
          alreadyVisited.add(nextOccurrence.getStart());
        }

        // double[] occurrenceSubsequence = extractSubsequence(series, nextOccurrence);

        double dist = normalizedDistance(series, currentEntry, nextOccurrence, zNormThreshold);
        distanceCalls++;

        // keep track of best so far distance
        if (dist < nearestNeighborDist) {
          nearestNeighborDist = dist;
          LOGGER.trace(" ** current NN at interval " + nextOccurrence.getStart() + "-"
              + nextOccurrence.getEnd() + ", distance: " + nearestNeighborDist);
        }
        if (dist < bestSoFarDistance) {
          LOGGER.trace(" ** abandoning the occurrences iterations");
          doRandomSearch = false;
          break;
        }
      }

      // check if we must continue with random neighbors
      if (doRandomSearch) {
        LOGGER.trace("starting random search");

        // init the visit array
        //
        int visitCounter = 0;
        int cIndex = 0;
        for (int j = 0; j < intervals.size(); j++) {
          RuleInterval interval = intervals.get(j);
          if (!(alreadyVisited.contains(interval.getStart()))) {
            visitArray[cIndex] = j;
            cIndex++;
          }
        }
        cIndex--;

        // shuffle the visit array
        //
        Random rnd = new Random(GrammarVizAnomaly.params.RANDOM_SEED);
        for (int j = cIndex; j > 0; j--) {
          int index = rnd.nextInt(j + 1);
          int a = visitArray[index];
          visitArray[index] = visitArray[j];
          visitArray[j] = a;
        }

        // while there are unvisited locations
        while (cIndex >= 0) {

          RuleInterval randomInterval = intervals.get(visitArray[cIndex]);
          cIndex--;

          // double[] randomSubsequence = extractSubsequence(series, randomInterval);

          double dist = normalizedDistance(series, currentEntry, randomInterval, zNormThreshold);
          distanceCalls++;

          // early abandoning of the search:
          // the current word is not discord, we have seen better
          if (dist < bestSoFarDistance) {
            nearestNeighborDist = dist;
            LOGGER.trace(" ** abandoning random visits loop, seen distance " + nearestNeighborDist
                + " at iteration " + visitCounter);
            break;
          }

          // keep track
          if (dist < nearestNeighborDist) {
            LOGGER.trace(" ** current NN id rule " + randomInterval.getId() + " at "
                + randomInterval.startPos + ", distance: " + dist);
            nearestNeighborDist = dist;
          }

          visitCounter = visitCounter + 1;

        } // while inner loop

      } // end of random search branch

      if (nearestNeighborDist > bestSoFarDistance) {
        LOGGER.trace(" updating discord candidate: rule " + currentEntry.getId() + " at "
            + currentEntry.getStart() + " len " + currentEntry.getLength() + " NN dist: "
            + bestSoFarDistance);
        bestSoFarDistance = nearestNeighborDist;
        bestSoFarPosition = currentEntry.getStart();
        bestSoFarLength = currentEntry.getLength();
        bestSoFarRule = currentEntry.getId();
      }

      LOGGER.trace(" . . iterated " + iterationCounter + " times, best distance:  "
          + bestSoFarDistance + " for a rule " + bestSoFarRule + " at " + bestSoFarPosition
          + " len " + bestSoFarLength);

    } // outer loop

    DiscordRecord res = new DiscordRecord(bestSoFarPosition, bestSoFarDistance,
        "pos,calls,len,rule " + bestSoFarPosition + " " + distanceCalls + " " + bestSoFarLength
            + " " + bestSoFarRule);

    res.setLength(bestSoFarLength);
    res.setRuleId(bestSoFarRule);
    res.setInfo("distance calls: " + distanceCalls);

    return res;
  }

  /**
   * Computes the normalized distance. The whole idea is that rules map to subsequences of different
   * length.
   *
   * @param series
   * @param reference
   * @param candidate
   * @param zNormThreshold
   * @return
   * @throws Exception
   */
  private static double normalizedDistance(double[] series, RuleInterval reference,
      RuleInterval candidate, double zNormThreshold) throws Exception {

    double[] ref = Arrays.copyOfRange(series, reference.getStart(), reference.getEnd());
    double[] cand = Arrays.copyOfRange(series, candidate.getStart(), candidate.getEnd());
    double divisor = Integer.valueOf(ref.length).doubleValue();

    // if the reference is the longest, we shrink it down with PAA
    //
    if (ref.length > cand.length) {
      ref = tp.paa(ref, cand.length);
      divisor = Integer.valueOf(cand.length).doubleValue(); // update the normalization value
    }
    // if the candidate is longest, we shrink it with PAA too
    //
    else {
      cand = tp.paa(cand, ref.length);
    }

    return ed.distance(tp.znorm(ref, zNormThreshold), tp.znorm(cand, zNormThreshold)) / divisor;

  }

  // /**
  // * Extracts a time series subsequence corresponding to the grammar rule adjusting for its
  // length.
  // *
  // * @param series
  // * @param randomInterval
  // * @return
  // */
  // private static double[] extractSubsequence(double[] series, RuleInterval randomInterval) {
  // return Arrays.copyOfRange(series, randomInterval.getStart(), randomInterval.getEnd());
  // }

  /**
   * Finds all the Sequitur rules with a given Id and populates their start and end into the array.
   *
   * @param id The rule Id.
   * @param intervals The rule intervals.
   * @return map of start - end.
   */
  private static ArrayList<Integer> listRuleOccurrences(int id, ArrayList<RuleInterval> intervals) {
    ArrayList<Integer> res = new ArrayList<Integer>(100);
    for (int j = 0; j < intervals.size(); j++) {
      RuleInterval i = intervals.get(j);
      if (id == i.getId()) {
        res.add(j);
      }
    }
    return res;
  }

  /**
   * Cloning an array.
   *
   * @param source the source array.
   * @return the clone.
   */
  private static ArrayList<RuleInterval> cloneIntervals(ArrayList<RuleInterval> source) {
    ArrayList<RuleInterval> res = new ArrayList<RuleInterval>(source.size());
    for (RuleInterval r : source) {
      res.add(new RuleInterval(r.getId(), r.getStart(), r.getEnd(), r.getCoverage()));
    }
    return res;
  }
}
