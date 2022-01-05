package net.seninp.tinker;

import java.util.List;

public class SamplerRecord {
  protected int window;
  protected int paa;
  protected int alphabet;
  protected double approxDist;
  protected int grammarSize;
  protected int grammarRules;
  protected int compressedGrammarSize;
  protected int prunedRules;
  protected int isCovered;

  protected double coverage;
  protected double reduction;

  public SamplerRecord(List<String> record) {
    int ctr = -1;
    window = Integer.valueOf(record.get(ctr + 1)).intValue();
    paa = Integer.valueOf(record.get(ctr + 2)).intValue();
    alphabet = Integer.valueOf(record.get(ctr + 3)).intValue();
    approxDist = Double.valueOf(record.get(ctr + 4)).doubleValue();
    grammarSize = Integer.valueOf(record.get(ctr + 5)).intValue();
    grammarRules = Integer.valueOf(record.get(ctr + 6)).intValue();
    compressedGrammarSize = Integer.valueOf(record.get(ctr + 7)).intValue();
    prunedRules = Integer.valueOf(record.get(ctr + 8)).intValue();
    isCovered = Integer.valueOf(record.get(ctr + 9)).intValue();
    coverage = Double.valueOf(record.get(ctr + 10)).intValue();
    reduction = (double) compressedGrammarSize / (double) grammarSize;
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + alphabet;
    long temp;
    temp = Double.doubleToLongBits(approxDist);
    result = prime * result + (int) (temp ^ (temp >>> 32));
    result = prime * result + compressedGrammarSize;
    temp = Double.doubleToLongBits(coverage);
    result = prime * result + (int) (temp ^ (temp >>> 32));
    result = prime * result + grammarRules;
    result = prime * result + grammarSize;
    result = prime * result + isCovered;
    result = prime * result + paa;
    result = prime * result + prunedRules;
    temp = Double.doubleToLongBits(reduction);
    result = prime * result + (int) (temp ^ (temp >>> 32));
    result = prime * result + window;
    return result;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj)
      return true;
    if (obj == null)
      return false;
    if (getClass() != obj.getClass())
      return false;
    SamplerRecord other = (SamplerRecord) obj;
    if (alphabet != other.alphabet)
      return false;
    if (Double.doubleToLongBits(approxDist) != Double.doubleToLongBits(other.approxDist))
      return false;
    if (compressedGrammarSize != other.compressedGrammarSize)
      return false;
    if (Double.doubleToLongBits(coverage) != Double.doubleToLongBits(other.coverage))
      return false;
    if (grammarRules != other.grammarRules)
      return false;
    if (grammarSize != other.grammarSize)
      return false;
    if (isCovered != other.isCovered)
      return false;
    if (paa != other.paa)
      return false;
    if (prunedRules != other.prunedRules)
      return false;
    if (Double.doubleToLongBits(reduction) != Double.doubleToLongBits(other.reduction))
      return false;
    if (window != other.window)
      return false;
    return true;
  }

  @Override
  public String toString() {
    return "SamplerRecord [window=" + window + ", paa=" + paa + ", alphabet=" + alphabet
        + ", approxDist=" + approxDist + ", grammarSize=" + grammarSize + ", grammarRules="
        + grammarRules + ", compressedGrammarSize=" + compressedGrammarSize + ", prunedRules="
        + prunedRules + ", isCovered=" + isCovered + ", coverage=" + coverage + ", reduction="
        + reduction + "]";
  }

}
