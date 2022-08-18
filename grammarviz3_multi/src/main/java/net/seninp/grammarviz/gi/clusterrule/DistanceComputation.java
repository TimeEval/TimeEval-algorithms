package net.seninp.grammarviz.gi.clusterrule;

import java.util.Random;

public class DistanceComputation {

  /**
   * Calculating the distance between time series and pattern.
   * 
   * @param ts a series of points for time series.
   * @param pValue a series of points for pattern.
   * @return the distance value.
   */
  protected double calcDistTSAndPattern(double[] ts, double[] pValue) {
    double INF = 10000000000000000000f;
    double bestDist = INF;
    int patternLen = pValue.length;

    int lastStartP = ts.length - pValue.length + 1;
    if (lastStartP < 1)
      return bestDist;

    Random rand = new Random();
    int startP = rand.nextInt((lastStartP - 1 - 0) + 1);

    double[] slidingWindow = new double[patternLen];

    System.arraycopy(ts, startP, slidingWindow, 0, patternLen);
    bestDist = eculideanDistNorm(pValue, slidingWindow);

    for (int i = 0; i < lastStartP; i++) {
      System.arraycopy(ts, i, slidingWindow, 0, patternLen);

      double tempDist = eculideanDistNormEAbandon(pValue, slidingWindow, bestDist);

      if (tempDist < bestDist) {
        bestDist = tempDist;
      }
    }

    return bestDist;
  }

  /**
   * Normalized early abandoned Euclidean distance.
   * 
   * @param ts1 the first series.
   * @param ts2 the second series.
   * @param bsfDist the distance value (used for early abandon).
   * @return the distance value.
   */
  protected double eculideanDistNormEAbandon(double[] ts1, double[] ts2, double bsfDist) {
    double dist = 0;
    double tsLen = ts1.length;

    double bsf = Math.pow(tsLen * bsfDist, 2);

    for (int i = 0; i < ts1.length; i++) {
      double diff = ts1[i] - ts2[i];
      dist += Math.pow(diff, 2);

      if (dist > bsf)
        return Double.NaN;

    }
    return Math.sqrt(dist) / tsLen;
  }

  /**
   * Normalized Euclidean distance.
   * 
   * @param ts1 the first series.
   * @param ts2 the second series.
   * @return the distance value.
   */
  protected double eculideanDistNorm(double[] ts1, double[] ts2) {
    double dist = 0;
    double tsLen = ts1.length;

    for (int i = 0; i < ts1.length; i++) {
      double diff = ts1[i] - ts2[i];
      dist += Math.pow(diff, 2);
    }

    return Math.sqrt(dist) / tsLen;
  }

}
