package net.seninp.grammarviz.anomaly;

import net.seninp.grammarviz.GrammarVizAnomaly;
import net.seninp.grammarviz.gi.logic.RuleInterval;
import net.seninp.jmotif.distance.EuclideanDistance;
import net.seninp.jmotif.sax.SAXProcessor;
import net.seninp.jmotif.sax.TSProcessor;
import net.seninp.jmotif.sax.discord.DiscordRecord;
import net.seninp.jmotif.sax.discord.DiscordRecords;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Implements RRA algorithm.
 *
 * @author psenin
 */
public class RRAImplementation {

    private static final TSProcessor tp = new TSProcessor();
    private static final EuclideanDistance ed = new EuclideanDistance();

    // static block - we instantiate the logger
    //
    private static final Logger LOGGER = LoggerFactory.getLogger(RRAImplementation.class);

    /**
     * Implements RRA -- an anomaly discovery algorithm based on discretization and grammar inference.
     * RRA stands for rare rule anomaly.
     *
     * @param series                The series to find discord at.
     * @param discordCollectionSize How many discords to find.
     * @param intervals             The intervals. In our implementation these come from the set of Sequitur
     *                              grammar rules.
     * @param zNormThreshold        - the normalization threshold (dstance).
     * @return Discords.
     * @throws TSException If error occurs.
     */
    public static DiscordRecords series2RRAAnomalies(double[] series, int discordCollectionSize,
                                                     List<RuleInterval> intervals, double zNormThreshold) throws Exception {

        Date gStart = new Date();

        // resulting discords collection
        DiscordRecords discords = new DiscordRecords();

        if (intervals.isEmpty()) {
            return discords;
        }

        // visit registry
        HashSet<Integer> registry = new HashSet<>(discordCollectionSize * intervals.get(0).getLength() * 2);

        // we conduct the search until the number of discords is less than desired
        //

        LOGGER.trace(
                "currently known discords: " + discords.getSize() + " out of " + discordCollectionSize);

        PriorityQueue<DiscordRecord> bestDiscords = findBestDiscordForIntervals(series, intervals, registry, zNormThreshold, discordCollectionSize);
        if (bestDiscords.size() == 0) {
            return discords;
        }

        while (bestDiscords.peek().getNNDistance() == Integer.MIN_VALUE || bestDiscords.peek().getPosition() == Integer.MIN_VALUE) {
            bestDiscords.remove();
        }

        // collect the result
        //
        for (DiscordRecord discord : bestDiscords) {
            discords.add(discord);
        }

        // mark the discord discovered
        /*
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
         */

        LOGGER.info(discords.getSize() + " discords found in "
                + SAXProcessor.timeToString(gStart.getTime(), new Date().getTime()));

        return discords;
    }

    /**
     * @param series
     * @param globalIntervals
     * @param registry
     * @param zNormThreshold
     * @return
     * @throws Exception
     */
    public static PriorityQueue<DiscordRecord> findBestDiscordForIntervals(double[] series,
                                                                           List<RuleInterval> globalIntervals, HashSet<Integer> registry, double zNormThreshold, int discordCollectionSize)
            throws Exception {

        // prepare the visits array, note that there can't be more points to visit that in a SAX index
        int[] visitArray = new int[globalIntervals.size()];

        // this is outer loop heuristics
        ArrayList<RuleInterval> intervals = cloneIntervals(globalIntervals);
        intervals.sort(Comparator.comparingDouble(RuleInterval::getCoverage));

        // init variables

        // we will iterate over words from rarest to frequent ones - this is an OUTER LOOP of the best
        // discord search
        //
        PriorityQueue<DiscordRecord> bestDiscords = new PriorityQueue<>(discordCollectionSize,
                Comparator.comparingDouble(DiscordRecord::getNNDistance));
        DiscordRecord dummyRecord = new DiscordRecord(-1, Integer.MIN_VALUE);
        bestDiscords.add(dummyRecord);
        //int[] registy = new int[series.length];


        LOGGER.trace("going to iterate over " + intervals.size() + " intervals looking for the discord");

        for (int i = 0; i < intervals.size(); i++) {

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
            HashSet<Integer> alreadyVisited = new HashSet<>(currentOccurences.size() + (markEnd - markStart));
            for (int j = markStart; j < markEnd; j++) {
                alreadyVisited.add(j);
            }

            // so, lets the search begin...
            double nearestNeighborDist = Double.MAX_VALUE;
            boolean doRandomSearch = true;

            // this is the first INNER LOOP
            for (Integer nextOccurrenceIdx : currentOccurences) {

                RuleInterval nextOccurrence = intervals.get(nextOccurrenceIdx);

                // skip the location we standing at, check if we overlap
                if (alreadyVisited.contains(nextOccurrence.getStart())) {
                    continue;
                } else {
                    alreadyVisited.add(nextOccurrence.getStart());
                }

                // double[] occurrenceSubsequence = extractSubsequence(series, nextOccurrence);

                double dist = normalizedDistance(series, currentEntry, nextOccurrence, zNormThreshold);

                // keep track of best so far distance
                if (dist < nearestNeighborDist) {
                    nearestNeighborDist = dist;
                    LOGGER.trace(" ** current NN at interval " + nextOccurrence.getStart() + "-"
                            + nextOccurrence.getEnd() + ", distance: " + nearestNeighborDist);
                }
                if (!shouldInsert(bestDiscords, dist, discordCollectionSize)) {
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

                    // early abandoning of the search:
                    // the current word is not discord, we have seen better
                    if (!shouldInsert(bestDiscords, dist, discordCollectionSize)) {
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

            if (nearestNeighborDist != Double.MAX_VALUE &&
                    shouldInsert(bestDiscords, nearestNeighborDist, discordCollectionSize)) {

                DiscordRecord potentialAnomaly = new DiscordRecord(currentEntry.getStart(), nearestNeighborDist);
                potentialAnomaly.setLength(currentEntry.getLength());
                potentialAnomaly.setRuleId(currentEntry.getId());
                if (bestDiscords.size() == discordCollectionSize) bestDiscords.remove();
                bestDiscords.add(potentialAnomaly);
            }

        } // outer loop

        return bestDiscords;
    }

    private static void addDiscord(DiscordRecord discord, int[] registry, PriorityQueue<DiscordRecord> bestDiscords, int discordCollectionSize) {
        if (bestDiscords.size() >= discordCollectionSize) {
            DiscordRecord removedDiscord = bestDiscords.remove();
            for (int i = removedDiscord.getPosition() - removedDiscord.getLength(); i < removedDiscord.getPosition() + removedDiscord.getLength(); i++) {

            }
        }
    }

    private static boolean shouldInsert(PriorityQueue<DiscordRecord> bestDiscords, double dist, int discordCollectionSize) {
        if (bestDiscords.size() < discordCollectionSize) {
            return true;
        }
        return dist > bestDiscords.peek().getNNDistance();
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
        double divisor = ref.length;

        // if the reference is the longest, we shrink it down with PAA
        //
        if (ref.length > cand.length) {
            ref = tp.paa(ref, cand.length);
            divisor = cand.length;
        }
        // if the candidate is longest, we shrink it with PAA too
        //
        else {
            cand = tp.paa(cand, ref.length);
        }
        double dist = ed.distance(tp.znorm(ref, zNormThreshold), tp.znorm(cand, zNormThreshold)) / divisor;
        if (dist > 5) {
            System.out.println("Shit");
        }
        return dist;

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
     * @param id        The rule Id.
     * @param intervals The rule intervals.
     * @return map of start - end.
     */
    private static ArrayList<Integer> listRuleOccurrences(int id, List<RuleInterval> intervals) {
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
    private static ArrayList<RuleInterval> cloneIntervals(List<RuleInterval> source) {
        ArrayList<RuleInterval> res = new ArrayList<RuleInterval>(source.size());
        for (RuleInterval r : source) {
            res.add(new RuleInterval(r.getId(), r.getStart(), r.getEnd(), r.getCoverage()));
        }
        return res;
    }
}
