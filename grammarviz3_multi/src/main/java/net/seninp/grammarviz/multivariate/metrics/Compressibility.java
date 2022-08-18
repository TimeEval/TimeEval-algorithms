package net.seninp.grammarviz.multivariate.metrics;

import net.seninp.grammarviz.gi.logic.GrammarRules;
import net.seninp.grammarviz.gi.sequitur.SequiturFactory;
import net.seninp.jmotif.sax.datastructure.SAXRecords;

public class Compressibility {

    private static double getNumPossibleUniqueWords(int tsLength, int windowSize) {
        return tsLength - windowSize + 1;
    }

    public static double getCompressibility(int numUniqueWords, int tsLength, int windowSize) {
        return 1 - (numUniqueWords / getNumPossibleUniqueWords(tsLength, windowSize));
    }

    public static double getJointCompressibility(double dim1,
                                                  double dim2,
                                                  double mergedDim) {
        return 2 * mergedDim / (dim1 + dim2);
    }

    public static double getSequiturCompressibility(SAXRecords discretizedTs,
                                                     int tsLength,
                                                     int windowSize,
                                                     int paaSize) {
        GrammarRules rules = SequiturFactory.discretized2SequiturRules(discretizedTs,
                windowSize, paaSize, tsLength);
        return 1 - ((double) rules.getRuleRecord(0).getRuleString().split(" ").length / getNumPossibleUniqueWords(tsLength, windowSize));
    }
}
