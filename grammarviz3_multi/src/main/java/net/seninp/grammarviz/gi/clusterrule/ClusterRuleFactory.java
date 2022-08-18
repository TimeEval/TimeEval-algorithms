package net.seninp.grammarviz.gi.clusterrule;

import java.util.ArrayList;

import net.seninp.grammarviz.gi.clusterrule.RuleOrganizer;
import net.seninp.grammarviz.gi.logic.GrammarRules;
import net.seninp.grammarviz.gi.logic.PackedRuleRecord;
import net.seninp.grammarviz.gi.logic.SameLengthMotifs;

public class ClusterRuleFactory {

  /**
   * Performs clustering.
   * 
   * @param ts the input time series.
   * @param grammarRules the grammar.
   * @param thresholdLength a parameter.
   * @param thresholdCom another parameter.
   * @param fractionTopDist yet another parameter.
   * 
   * @return pruned ruleset.
   */
  public static ArrayList<SameLengthMotifs> performPruning(double[] ts, GrammarRules grammarRules,
      double thresholdLength, double thresholdCom, double fractionTopDist) {

    RuleOrganizer ro = new RuleOrganizer();

    ArrayList<SameLengthMotifs> allClassifiedMotifs = ro.classifyMotifs(thresholdLength,
        grammarRules);

    allClassifiedMotifs = ro.removeOverlappingInSimiliar(allClassifiedMotifs, grammarRules, ts,
        thresholdCom);

    ArrayList<SameLengthMotifs> newAllClassifiedMotifs = ro.refinePatternsByClustering(grammarRules,
        ts, allClassifiedMotifs, fractionTopDist);

    return newAllClassifiedMotifs;
  }

  /**
   * Gets packed rules set.
   * 
   * @param newAllClassifiedMotifs a parameter.
   * @return packed rule set.
   */
  public static ArrayList<PackedRuleRecord> getPackedRule(
      ArrayList<SameLengthMotifs> newAllClassifiedMotifs) {

    ArrayList<PackedRuleRecord> arrPackedRuleRecords = new ArrayList<PackedRuleRecord>();
    int i = 0;
    for (SameLengthMotifs subsequencesInClass : newAllClassifiedMotifs) {
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

    return arrPackedRuleRecords;
  }
}
