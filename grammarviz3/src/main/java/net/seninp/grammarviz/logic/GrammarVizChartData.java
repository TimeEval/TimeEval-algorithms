package net.seninp.grammarviz.logic;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Observable;
import java.util.Observer;
import java.util.Random;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import com.apporiented.algorithm.clustering.AverageLinkageStrategy;
import com.apporiented.algorithm.clustering.Cluster;
import com.apporiented.algorithm.clustering.ClusteringAlgorithm;
import com.apporiented.algorithm.clustering.DefaultClusteringAlgorithm;
import net.seninp.gi.logic.GrammarRuleRecord;
import net.seninp.gi.logic.GrammarRules;
import net.seninp.gi.logic.RuleInterval;
import net.seninp.gi.rulepruner.RulePrunerFactory;
import net.seninp.grammarviz.model.GrammarVizMessage;
import net.seninp.jmotif.sax.NumerosityReductionStrategy;
import net.seninp.jmotif.sax.discord.DiscordRecords;
import net.seninp.grammarviz.GrammarVizAnomaly;

/**
 * The main data structure used in SAXSequitur. It contains all the information needed for charting
 * and tables.
 *
 * @author Manfred Lerner, seninp
 *
 */
public class GrammarVizChartData extends Observable implements Observer {

  /** SAX conversion parameters. */
  protected final boolean slidingWindowOn;
  protected final NumerosityReductionStrategy numerosityReductionStrategy;
  protected final int saxWindowSize;
  protected final int saxAlphabetSize;
  protected final int saxPAASize;
  protected final double zNormThreshold;

  /** Original data file name. */
  @SuppressWarnings("unused")
  private final String inputFname;

  /** Original data which will be used for the chart. */
  protected final double[] originalTimeSeries;

  /** The whole timeseries as a string */
  private String saxDisplayString = null;

  /** The grammar rules. */
  private GrammarRules grammarRules;

  /** The discords. */
  protected DiscordRecords discords;

  /** Pruning related vars. */
  private SAXPointsNumber[] pointsNumberRemoveStrategy;
  private ArrayList<SameLengthMotifs> allClassifiedMotifs;
  private ArrayList<PackedRuleRecord> arrPackedRuleRecords;

  /**
   * Constructor.
   *
   * @param dataFileName the original filename.
   * @param ts the time series.
   * @param useSlidingWindow
   * @param numerosityReductionStrategy
   * @param windowSize SAX window size.
   * @param alphabetSize SAX alphabet size.
   * @param paaSize SAX PAA size.
   * @param zNormThreshold the z-normalization threshold.
   */
  public GrammarVizChartData(String dataFileName, double[] ts, boolean useSlidingWindow,
      NumerosityReductionStrategy numerosityReductionStrategy, int windowSize, int paaSize,
      int alphabetSize, double zNormThreshold) {

    this.inputFname = dataFileName;

    this.slidingWindowOn = useSlidingWindow;
    this.numerosityReductionStrategy = numerosityReductionStrategy;

    this.originalTimeSeries = ts;

    this.saxWindowSize = windowSize;
    this.saxPAASize = paaSize;
    this.saxAlphabetSize = alphabetSize;
    this.zNormThreshold = zNormThreshold;
  }

  /**
   * Get the original, untransformed time series.
   *
   * @return the original time series
   */
  public double[] getOriginalTimeseries() {
    return originalTimeSeries;
  }

  /**
   * Sets the grammar rules data.
   *
   * @param rules the grammar rules collection.
   */
  public void setGrammarRules(GrammarRules rules) {
    this.grammarRules = rules;
  }

  /**
   * Get the grammar rules.
   *
   * @return the grammar rules collection.
   */
  public GrammarRules getGrammarRules() {
    return this.grammarRules;
  }

  /**
   * @return SAX window size
   */
  public int getSAXWindowSize() {
    return saxWindowSize;
  }

  /**
   * @return SAX alphabet size
   */
  public int getSAXAlphabetSize() {
    return saxAlphabetSize;
  }

  /**
   * @return SAX PAA size
   */
  public int getSAXPaaSize() {
    return saxPAASize;
  }

  public boolean isSlidingWindowOn() {
    return this.slidingWindowOn;
  }

  public double getZNormThreshold() {
    return this.zNormThreshold;
  }

  /**
   * Get the collection of transformed rule records.
   *
   * @return the collection of transformed rules.
   */
  public ArrayList<PackedRuleRecord> getArrPackedRuleRecords() {
    return arrPackedRuleRecords;
  }

  /**
   * Set the collection of transformed rule records.
   *
   * @param arrPackedRuleRecords the collection of transformed rules.
   */
  public void setArrPackedRuleRecords(ArrayList<PackedRuleRecord> arrPackedRuleRecords) {
    this.arrPackedRuleRecords = arrPackedRuleRecords;
  }

  /**
   * Сonverts rules from a foreign alphabet to the internal original SAX alphabet.
   *
   * @param rule the SAX rule in foreign SAX alphabet.
   * @return the SAX string in original alphabet, e.g. aabbdd.
   */
  public String convert2OriginalSAXAlphabet(char firstForeignAlphabetChar, String rule) {
    String textRule = rule;
    for (int i = 0; i < getSAXAlphabetSize(); i++) {
      char c1 = (char) (firstForeignAlphabetChar + i);
      char c2 = (char) ('a' + i);
      textRule = textRule.replace(c1, c2);
    }
    return textRule;
  }

  /**
   * @param SAXDisplay SAX display formatted string
   */
  public void setSAXDisplay(String SAXDisplay) {
    saxDisplayString = SAXDisplay;
  }

  /**
   * @return SAX display formatted string.
   */
  public String getSAXDisplay() {
    return saxDisplayString;
  }

  /**
   * Recovers start and stop coordinates ofRule's subsequences.
   *
   * @param ruleIdx The rule index.
   * @return The array of all intervals corresponding to this rule.
   */
  public ArrayList<RuleInterval> getRulePositionsByRuleNum(Integer ruleIdx) {
    GrammarRuleRecord ruleRec = this.grammarRules.getRuleRecord(ruleIdx);
    return ruleRec.getRuleIntervals();
  }

  /**
   * Get the rule-corresponding subsequences from a class.
   *
   * @param clsIdx the class index.
   * @return the class-associated subsequences.
   */
  public ArrayList<RuleInterval> getSubsequencesPositionsByClassNum(Integer clsIdx) {

    // this will be the result
    ArrayList<RuleInterval> positions = new ArrayList<RuleInterval>();

    // the sub-sequences class container
    SameLengthMotifs thisClass = allClassifiedMotifs.get(clsIdx);

    // Use minimal length to name the file.
    // String fileName = thisClass.getMinMotifLen() + ".txt";
    // The position of those sub-sequences in the original time series.
    // String positionFileName = thisClass.getMinMotifLen() + "Position" + ".txt";

    // String path = "Result" + System.getProperties().getProperty("file.separator") + "data"
    // + System.getProperties().getProperty("file.separator");

    double[] values = this.getOriginalTimeseries();

    XYSeriesCollection data = new XYSeriesCollection();

    for (SAXMotif subSequence : thisClass.getSameLenMotifs()) {
      positions.add(new RuleInterval(subSequence.getPos().startPos, subSequence.getPos().endPos));
    }

    int index = 0;
    for (RuleInterval pos : positions) {
      XYSeries dataset = new XYSeries("Daten" + String.valueOf(index));

      int start = pos.getStart();
      int end = pos.getEnd() - 1;

      int count = 0;
      for (int i = start; (i <= end) && (i < values.length); i++) {
        dataset.add(count++, values[i]);
      }
      data.addSeries(dataset);
      index++;
    }
    // SAXFileIOHelper.writeFileXYSeries(path, fileName, positionFileName, data, positions);

    return positions;
  }

  public int getRulesNumber() {
    return grammarRules.size();
  }

  // ********************************
  // Refactoring in Xing's code below
  // ********************************

  public GrammarRuleRecord getRule(Integer ruleIndex) {
    return this.grammarRules.get(ruleIndex);
  }

  /**
   * Performs greedy rule prunung, the grammar will be lost.
   */
  public void performRulePruning() {
    GrammarRules prunedRulesSet = RulePrunerFactory.performPruning(this.originalTimeSeries,
        this.grammarRules);
    this.grammarRules = prunedRulesSet;
  }

  /**
   * This computes anomalies.
   *
   * @throws Exception
   */
  public void findAnomalies() throws Exception {
    GrammarVizAnomalyFinder finder = new GrammarVizAnomalyFinder(this);
    finder.addObserver(this);
    finder.run();
  }

  public DiscordRecords getAnomalies() {
    return this.discords;
  }

  /**
   * This method counts how many times each data point is used in ANY sequitur rule (i.e. data point
   * 1 appears only in R1 and R2, the number for data point 1 is two). The function will get the
   * occurrence time for all points, and write the result into a text file named as
   * "PointsNumber.txt".
   */
  protected void countPointNumber() {

    // init the data structure and copy the original values
    SAXPointsNumber pointsNumber[] = new SAXPointsNumber[this.originalTimeSeries.length];
    for (int i = 0; i < this.originalTimeSeries.length; i++) {
      pointsNumber[i] = new SAXPointsNumber();
      pointsNumber[i].setPointIndex(i);
      pointsNumber[i].setPointValue(this.originalTimeSeries[i]);
    }

    // get all the rules and populate the occurrence density
    int rulesNum = this.getRulesNumber();
    for (int i = 0; i < rulesNum; i++) {
      ArrayList<RuleInterval> arrPos = this.getRulePositionsByRuleNum(i);
      for (RuleInterval saxPos : arrPos) {
        int start = saxPos.getStart();
        int end = saxPos.getEnd() - 1;
        for (int position = start; position <= end; position++) {
          pointsNumber[position]
              .setPointOccurenceNumber(pointsNumber[position].getPointOccurenceNumber() + 1);
        }
      }
    }

    // make an output
    // String path = "Result" + System.getProperties().getProperty("file.separator");
    // String fileName = "PointsNumber.txt";
    // SAXFileIOHelper.deleteFile(path, fileName);
    // SAXFileIOHelper.writeFile(path, fileName, Arrays.toString(pointsNumber));

    this.pointsNumberRemoveStrategy = pointsNumber;
  }

  /**
   * This method counts how many times each data point is used in REDUCED sequitur rule (i.e. data
   * point 1 appears only in R1 and R2, the number for data point 1 is two). The function will get
   * the occurrence time for all points, and write the result into a text file named as
   * "PointsNumberAfterRemoving.txt".
   */
  protected void countPointNumberAfterRemoving() {

    // init the data structure and copy the original values
    SAXPointsNumber pointsNumber[] = new SAXPointsNumber[this.originalTimeSeries.length];
    for (int i = 0; i < this.originalTimeSeries.length; i++) {
      pointsNumber[i] = new SAXPointsNumber();
      pointsNumber[i].setPointIndex(i);
      pointsNumber[i].setPointValue(this.originalTimeSeries[i]);
    }

    for (SameLengthMotifs sameLenMotifs : this.getReducedMotifs()) {
      for (SAXMotif motif : sameLenMotifs.getSameLenMotifs()) {
        RuleInterval pos = motif.getPos();
        for (int i = pos.getStart(); i <= pos.getEnd() - 1; i++) {
          pointsNumber[i].setPointOccurenceNumber(pointsNumber[i].getPointOccurenceNumber() + 1);
          // pointsNumber[i].setRule(textRule);
        }
      }
    }

    // make an output
    // String path = "Result" + System.getProperties().getProperty("file.separator");
    // String fileName = "PointsNumberAfterRemoving.txt";
    // SAXFileIOHelper.deleteFile(path, fileName);
    // SAXFileIOHelper.writeFile(path, fileName, Arrays.toString(pointsNumber));

  }

  /**
   * Cleans-up the rules set by classifying the sub-sequences by length and removing the overlapping
   * in the same length range.
   *
   * Sub-sequences with the length difference within threshold: "thresouldLength" will be classified
   * as a class with the function "classifyMotifs(double)", i.e. 1-100 and 101-205 will be
   * classified as a class when the threshold is 0.1, because the length difference is 5, which is
   * less than the threshold (0.1 * 100 = 10). If two sub-sequences within one class share a common
   * part which is more than the threshold: "thresouldCom", one of them will be removed by the
   * function "removeOverlappingInSimiliar(double)". i.e. 1-100 and 21-120.
   *
   * @param intraThreshold, the threshold between the same motifs.
   * @param interThreshould, the threshold between the different motifs.
   */
  protected void removeOverlapping(double intraThreshold, double interThreshould) {

    classifyMotifs(intraThreshold);
    // ArrayList<SAXMotif> motifsBeDeleted = removeOverlappingInSimiliar(interThreshould);

    // String path = "Result" +
    // System.getProperties().getProperty("file.separator");
    // String fileName = "Deleted Motifs.txt";
    // SAXFileIOHelper.deleteFile(path, fileName);
    // SAXFileIOHelper.writeFile(path, fileName, motifsBeDeleted.toString());

  }

  /**
   * Classify the motifs based on their length.
   *
   * It calls "getAllMotifs()" to get all the sub-sequences that were generated by Sequitur rules in
   * ascending order. Then bins all the sub-sequences by length based on the length of the first
   * sub-sequence in each class, that is, the shortest sub-sequence in each class.
   *
   * @param lengthThreshold the motif length threshold.
   */
  protected void classifyMotifs(double lengthThreshold) {

    // reset vars
    allClassifiedMotifs = new ArrayList<SameLengthMotifs>();

    // down to business
    ArrayList<SAXMotif> allMotifs = getAllMotifs();

    // is this one better?
    int currentIndex = 0;
    for (SAXMotif tmpMotif : allMotifs) {

      currentIndex++;

      if (tmpMotif.isClassified()) {
        // this breaks the loop flow, so it goes to //for (SAXMotif tempMotif : allMotifs) {
        continue;
      }

      SameLengthMotifs tmpSameLengthMotifs = new SameLengthMotifs();
      int tmpMotifLen = tmpMotif.getPos().getEnd() - tmpMotif.getPos().getStart() + 1;
      int minLen = tmpMotifLen;
      int maxLen = tmpMotifLen;

      // TODO: assuming that this motif has not been processed, right?
      ArrayList<SAXMotif> newMotifClass = new ArrayList<SAXMotif>();
      newMotifClass.add(tmpMotif);
      tmpMotif.setClassified(true);

      // TODO: this motif assumed to be the first one of it's class, traverse the rest down
      for (int i = currentIndex; i < allMotifs.size(); i++) {

        SAXMotif anotherMotif = allMotifs.get(i);

        // if the two motifs are similar or not.
        int anotherMotifLen = anotherMotif.getPos().getEnd() - anotherMotif.getPos().getStart() + 1;

        // if they have the similar length.
        if (Math.abs(anotherMotifLen - tmpMotifLen) < (tmpMotifLen * lengthThreshold)) {
          newMotifClass.add(anotherMotif);
          anotherMotif.setClassified(true);
          if (anotherMotifLen > maxLen) {
            maxLen = anotherMotifLen;
          }
          else if (anotherMotifLen < minLen) {
            minLen = anotherMotifLen;
          }
        }
      }

      tmpSameLengthMotifs.setSameLenMotifs(newMotifClass);
      tmpSameLengthMotifs.setMinMotifLen(minLen);
      tmpSameLengthMotifs.setMaxMotifLen(maxLen);
      allClassifiedMotifs.add(tmpSameLengthMotifs);
    }
    // System.out.println();
  }

  protected ArrayList<SAXMotif> removeOverlappingInSimiliar(double thresouldCom) {

    ArrayList<SAXMotif> motifsBeDeleted = new ArrayList<SAXMotif>();

    countPointNumber();
    for (SameLengthMotifs sameLenMotifs : allClassifiedMotifs) {
      outer: for (int j = 0; j < sameLenMotifs.getSameLenMotifs().size(); j++) {
        SAXMotif tempMotif = sameLenMotifs.getSameLenMotifs().get(j);
        int tempMotifLen = tempMotif.getPos().getEnd() - tempMotif.getPos().getStart() + 1;

        for (int i = j + 1; i < sameLenMotifs.getSameLenMotifs().size(); i++) {
          SAXMotif anotherMotif = sameLenMotifs.getSameLenMotifs().get(i);
          int anotherMotifLen = anotherMotif.getPos().getEnd() - anotherMotif.getPos().getStart()
              + 1;

          double minEndPos = Math.min(tempMotif.getPos().getEnd(), anotherMotif.getPos().getEnd());
          double maxStartPos = Math.max(tempMotif.getPos().getStart(),
              anotherMotif.getPos().getStart());
          // the length in common.
          double commonLen = minEndPos - maxStartPos + 1;

          // if they are overlapped motif, remove the shorter one
          if (commonLen > (tempMotifLen * thresouldCom)) {
            SAXMotif deletedMotif = new SAXMotif();
            SAXMotif similarWith = new SAXMotif();

            boolean isAnotherBetter;

            if (pointsNumberRemoveStrategy != null) {
              isAnotherBetter = decideRemove(anotherMotif, tempMotif);
            }
            else {
              isAnotherBetter = anotherMotifLen > tempMotifLen;

            }
            if (isAnotherBetter) {
              deletedMotif = tempMotif;
              similarWith = anotherMotif;
              sameLenMotifs.getSameLenMotifs().remove(j);
              deletedMotif.setSimilarWith(similarWith);
              motifsBeDeleted.add(deletedMotif);
              j--;
              continue outer;
            }
            else {
              deletedMotif = anotherMotif;
              similarWith = tempMotif;
              sameLenMotifs.getSameLenMotifs().remove(i);
              deletedMotif.setSimilarWith(similarWith);
              motifsBeDeleted.add(deletedMotif);
              i--;
            }
          }
        }
      }

      int minLength = sameLenMotifs.getSameLenMotifs().get(0).getPos().endPos
          - sameLenMotifs.getSameLenMotifs().get(0).getPos().startPos + 1;
      int sameLenMotifsSize = sameLenMotifs.getSameLenMotifs().size();
      int maxLength = sameLenMotifs.getSameLenMotifs().get(sameLenMotifsSize - 1).getPos().endPos
          - sameLenMotifs.getSameLenMotifs().get(sameLenMotifsSize - 1).getPos().startPos + 1;
      sameLenMotifs.setMinMotifLen(minLength);
      sameLenMotifs.setMaxMotifLen(maxLength);
    }
    countPointNumberAfterRemoving();

    refinePatternsByClustering();
    return motifsBeDeleted;
  }

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

  protected double eculideanDistNorm(double[] ts1, double[] ts2) {
    double dist = 0;
    double tsLen = ts1.length;

    for (int i = 0; i < ts1.length; i++) {
      double diff = ts1[i] - ts2[i];
      dist += Math.pow(diff, 2);
    }

    return Math.sqrt(dist) / tsLen;
  }

  /**
   * Calculating the distance between time series and pattern.
   *
   * @param ts , a series of points for time series.
   * @param pValue , a series of points for pattern.
   * @return
   */
  protected double calcDistTSAndPattern(double[] ts, double[] pValue) {
    double INF = 10000000000000000000f;
    double bestDist = INF;
    int patternLen = pValue.length;

    int lastStartP = ts.length - pValue.length + 1;
    if (lastStartP < 1)
      return bestDist;

    Random rand = new Random(GrammarVizAnomaly.params.RANDOM_SEED);
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

  protected void refinePatternsByClustering() {
    double[] origTS = originalTimeSeries;
    ArrayList<SameLengthMotifs> newAllClassifiedMotifs = new ArrayList<SameLengthMotifs>();
    for (SameLengthMotifs sameLenMotifs : allClassifiedMotifs) {
      ArrayList<RuleInterval> arrPos = new ArrayList<RuleInterval>();
      ArrayList<SAXMotif> subsequences = sameLenMotifs.getSameLenMotifs();
      for (SAXMotif ss : subsequences) {
        arrPos.add(ss.getPos());
      }

      int patternNum = arrPos.size();
      if (patternNum < 2) {
        continue;
      }
      double dt[][] = new double[patternNum][patternNum];
      // Build distance matrix.
      for (int i = 0; i < patternNum; i++) {
        RuleInterval saxPos = arrPos.get(i);

        int start1 = saxPos.getStart();
        int end1 = saxPos.getEnd();
        double[] ts1 = Arrays.copyOfRange(origTS, start1, end1);

        for (int j = 0; j < arrPos.size(); j++) {
          RuleInterval saxPos2 = arrPos.get(j);
          if (dt[i][j] > 0) {
            continue;
          }
          double d = 0;
          dt[i][j] = d;
          if (i == j) {
            continue;
          }
          int start2 = saxPos2.getStart();
          int end2 = saxPos2.getEnd();
          double[] ts2 = Arrays.copyOfRange(origTS, start2, end2);

          if (ts1.length > ts2.length)
            d = calcDistTSAndPattern(ts1, ts2);
          else
            d = calcDistTSAndPattern(ts2, ts1);

          // DTW dtw = new DTW(ts1, ts2);
          // d = dtw.warpingDistance;

          dt[i][j] = d;
        }
      }

      String[] patternsName = new String[patternNum];
      for (int i = 0; i < patternNum; i++) {
        patternsName[i] = String.valueOf(i);
      }

      ClusteringAlgorithm alg = new DefaultClusteringAlgorithm();
      Cluster cluster = alg.performClustering(dt, patternsName, new AverageLinkageStrategy());

      // int minPatternPerCls = (int) (0.3 * patternNum);
      // minPatternPerCls = minPatternPerCls > 0 ? minPatternPerCls : 1;
      int minPatternPerCls = 1;

      if (cluster.getDistance() == null) {
        // System.out.print(false);
        continue;
      }

      // TODO: refine hard coded threshold
      // double cutDist = cluster.getDistance() * 0.67;
      double cutDist = cluster.getDistanceValue() * 0.67;

      ArrayList<String[]> clusterTSIdx = findCluster(cluster, cutDist, minPatternPerCls);
      while (clusterTSIdx.size() <= 0) {
        cutDist += cutDist / 2;
        clusterTSIdx = findCluster(cluster, cutDist, minPatternPerCls);
      }

      newAllClassifiedMotifs.addAll(SeparateMotifsByClustering(clusterTSIdx, sameLenMotifs));
    }
    allClassifiedMotifs = newAllClassifiedMotifs;
  }

  private ArrayList<SameLengthMotifs> SeparateMotifsByClustering(ArrayList<String[]> clusterTSIdx,
      SameLengthMotifs sameLenMotifs) {
    ArrayList<SameLengthMotifs> newResult = new ArrayList<SameLengthMotifs>();
    if (clusterTSIdx.size() > 1) {
      ArrayList<SAXMotif> subsequences = sameLenMotifs.getSameLenMotifs();
      for (String[] idxesInCluster : clusterTSIdx) {
        SameLengthMotifs newIthSLM = new SameLengthMotifs();
        ArrayList<SAXMotif> sameLenSS = new ArrayList<SAXMotif>();
        int minL = sameLenMotifs.getMinMotifLen();
        int maxL = sameLenMotifs.getMaxMotifLen();

        for (String i : idxesInCluster) {
          SAXMotif ssI = subsequences.get(Integer.parseInt(i));
          int len = ssI.getPos().getEnd() - ssI.getPos().getStart();
          if (len < minL) {
            minL = len;
          }
          else if (len > maxL) {
            maxL = len;
          }
          sameLenSS.add(ssI);
        }

        newIthSLM.setSameLenMotifs(sameLenSS);
        newIthSLM.setMaxMotifLen(maxL);
        newIthSLM.setMinMotifLen(minL);
        newResult.add(newIthSLM);
      }
    }
    else {
      newResult.add(sameLenMotifs);
    }

    return newResult;
  }

  private ArrayList<String[]> findCluster(Cluster cluster, double cutDist, int minPatternPerCls) {

    ArrayList<String[]> clusterTSIdx = new ArrayList<String[]>();

    if (cluster.getDistance() != null) {
      // if (cluster.getDistance() > cutDist) {
      if (cluster.getDistanceValue() > cutDist) {
        if (cluster.getChildren().size() > 0) {
          clusterTSIdx.addAll(findCluster(cluster.getChildren().get(0), cutDist, minPatternPerCls));
          clusterTSIdx.addAll(findCluster(cluster.getChildren().get(1), cutDist, minPatternPerCls));
        }
      }
      else {
        // String[] idxes = cluster.getName().split("&");
        ArrayList<String> itemsInCluster = getNameInCluster(cluster);
        String[] idxes = itemsInCluster.toArray(new String[itemsInCluster.size()]);
        if (idxes.length > minPatternPerCls) {
          clusterTSIdx.add(idxes);
        }
      }
    }

    return clusterTSIdx;
  }

  private ArrayList<String> getNameInCluster(Cluster cluster) {
    ArrayList<String> itemsInCluster = new ArrayList<String>();

    String nodeName;
    if (cluster.isLeaf()) {
      nodeName = cluster.getName();
      itemsInCluster.add(nodeName);
    }
    else {
      // String[] clusterName = cluster.getName().split("#");
      // nodeName = clusterName[1];
    }

    for (Cluster child : cluster.getChildren()) {
      ArrayList<String> childrenNames = getNameInCluster(child);
      itemsInCluster.addAll(childrenNames);
    }
    return itemsInCluster;
  }

  /**
   * Stores all the sub-sequences that generated by Sequitur rules into an array list sorted by
   * sub-sequence length in ascending order.
   *
   * @return the list of all sub-sequences sorted by length in ascending order.
   */
  protected ArrayList<SAXMotif> getAllMotifs() {

    // result
    ArrayList<SAXMotif> allMotifs = new ArrayList<SAXMotif>();

    // iterate over all rules
    for (int i = 0; i < this.getRulesNumber(); i++) {

      // iterate over all segments/motifs/sub-sequences which correspond to the rule
      ArrayList<RuleInterval> arrPos = this.getRulePositionsByRuleNum(i);
      for (RuleInterval saxPos : arrPos) {
        SAXMotif motif = new SAXMotif();
        motif.setPos(saxPos);
        motif.setRuleIndex(i);
        motif.setClassified(false);
        allMotifs.add(motif);
      }

    }

    // ascending order
    Collections.sort(allMotifs);
    return allMotifs;
  }

  /**
   * Decide which one from overlapping subsequences should be removed. The decision rule is that
   * each sub-sequence has a weight, the one with the smaller weight should be removed.
   *
   * The weight is S/(A * L). S is the sum of occurrence time of all data points in that
   * sub-sequence, A is the average weight of the whole time series, and L is the length of that
   * sub-sequence.
   *
   * @param motif1
   * @param motif2
   *
   * @return
   */
  protected boolean decideRemove(SAXMotif motif1, SAXMotif motif2) {

    // motif1 details
    int motif1Start = motif1.getPos().getStart();
    int motif1End = motif1.getPos().getEnd() - 1;
    int length1 = motif1End - motif1Start;

    // motif2 details
    int motif2Start = motif2.getPos().getStart();
    int motif2End = motif1.getPos().getEnd() - 1;
    int length2 = motif2End - motif2Start;

    int countsMotif1 = 0;
    int countsMotif2 = 0;

    // compute the averageWeight
    double averageWeight = 1;
    int count = 0;
    for (int i = 0; i < pointsNumberRemoveStrategy.length; i++) {
      count += pointsNumberRemoveStrategy[i].getPointOccurenceNumber();
    }
    averageWeight = (double) count / (double) pointsNumberRemoveStrategy.length;

    // compute counts for motif 1
    for (int i = motif1Start; i <= motif1End; i++) {
      countsMotif1 += pointsNumberRemoveStrategy[i].getPointOccurenceNumber();
    }

    // compute counts for motif 2
    for (int i = motif2Start; i <= motif2End; i++) {
      countsMotif2 += pointsNumberRemoveStrategy[i].getPointOccurenceNumber();
    }

    // get weights
    double weight1 = countsMotif1 / (averageWeight * length1);
    double weight2 = countsMotif2 / (averageWeight * length2);

    if (weight1 > weight2) {
      return true;
    }

    return false;
  }

  /**
   * Performs rules pruning based on their overlap.
   *
   * @param thresholdLength
   * @param thresholdCom
   */
  public void performRemoveOverlapping(double thresholdLength, double thresholdCom) {

    removeOverlapping(thresholdLength, thresholdCom);

    arrPackedRuleRecords = new ArrayList<PackedRuleRecord>();

    int i = 0;
    for (SameLengthMotifs subsequencesInClass : allClassifiedMotifs) {
      int classIndex = i;
      int subsequencesNumber = subsequencesInClass.getSameLenMotifs().size();
      int minLength = subsequencesInClass.getMinMotifLen();
      int maxLength = subsequencesInClass.getMaxMotifLen();

      PackedRuleRecord packedRuleRecord = new PackedRuleRecord();
      packedRuleRecord.setClassIndex(classIndex);
      packedRuleRecord.setSubsequenceNumber(subsequencesNumber);
      packedRuleRecord.setMinLength(minLength);
      packedRuleRecord.setMaxLength(maxLength);

      arrPackedRuleRecords.add(packedRuleRecord);
      i++;
    }

  }

  public ArrayList<SameLengthMotifs> getReducedMotifs() {
    // TODO Auto-generated method stub
    return allClassifiedMotifs;
  }

  @Override
  public void update(Observable o, Object arg) {
    if (arg instanceof GrammarVizMessage) {
      this.setChanged();
      notifyObservers(arg);
    }
  }

  @SuppressWarnings("unused")
  private double getPeriodError(int[] starts, double meanPeriod) {
    double sqd = 0.0;
    for (int i = 1; i < starts.length; i++) {
      double periodDiff = ((double) starts[i] - starts[i - 1]) - meanPeriod;
      sqd = sqd + periodDiff * periodDiff;
    }
    return Math.sqrt(sqd / (starts.length - 1));
  }

  @SuppressWarnings("unused")
  private double getMeanPeriod(int[] starts) {
    int sum = 0;
    for (int i = 1; i < starts.length; i++) {
      sum = sum + starts[i] - starts[i - 1];
    }
    return ((double) sum) / (double) (starts.length - 1);
  }

  @SuppressWarnings("unused")
  private Integer getMeanLength(int[] lengths) {
    int sum = 0;
    for (int l : lengths) {
      sum = sum + l;
    }
    return sum / lengths.length;
  }

}
