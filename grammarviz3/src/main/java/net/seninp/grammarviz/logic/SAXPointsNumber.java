package net.seninp.grammarviz.logic;

public class SAXPointsNumber {
  private int pointIndex;
  private double pointValue;
  private int pointOccurenceNumber = 0;

  // private String rule;

  /**
   * @return the pointIndex
   */
  public int getPointIndex() {
    return pointIndex;
  }

  /**
   * @param pointIndex the pointIndex to set
   */
  public void setPointIndex(int pointIndex) {
    this.pointIndex = pointIndex;
  }

  /**
   * @return the pointValue
   */
  public double getPointValue() {
    return pointValue;
  }

  /**
   * @param pointValue the pointValue to set
   */
  public void setPointValue(double pointValue) {
    this.pointValue = pointValue;
  }

  /**
   * @return the pointOccurenceNumber
   */
  public int getPointOccurenceNumber() {
    return pointOccurenceNumber;
  }

  /**
   * @param pointOccurenceNumber the pointOccurenceNumber to set
   */
  public void setPointOccurenceNumber(int pointOccurenceNumber) {
    this.pointOccurenceNumber = pointOccurenceNumber;
  }

  /**
   * @return the rule
   */
  // public String getRule() {
  // return rule;
  // }

  /**
   * @param rule the rule to set
   */
  // public void setRule(String rule) {
  // this.rule = rule;
  // }

  public String toString() {
    return "Index: " + pointIndex + "\tValue: " + pointValue + "\tNumber of Occurrence: "
        + pointOccurenceNumber + "\n\n";
  }
}
