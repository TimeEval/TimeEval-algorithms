package net.seninp.grammarviz.logic;

public enum CoverageCountStrategy {
  COUNT(0), LEVEL(1), OCCURRENCE(2), YIELD(3), PRODUCT(4);

  private final int index;

  CoverageCountStrategy(int index) {
    this.index = index;
  }

  public int index() {
    return index;
  }

  public static CoverageCountStrategy fromValue(int value) {
    switch (value) {
    case 0:
      return CoverageCountStrategy.COUNT;
    case 1:
      return CoverageCountStrategy.LEVEL;
    case 2:
      return CoverageCountStrategy.OCCURRENCE;
    case 3:
      return CoverageCountStrategy.YIELD;
    case 4:
      return CoverageCountStrategy.PRODUCT;
    default:
      throw new RuntimeException("Unknown index:" + value);
    }
  }

  public static CoverageCountStrategy fromValue(String value) {
    if (value.equalsIgnoreCase("count")) {
      return CoverageCountStrategy.COUNT;
    }
    else if (value.equalsIgnoreCase("level")) {
      return CoverageCountStrategy.LEVEL;
    }
    else if (value.equalsIgnoreCase("occurrence")) {
      return CoverageCountStrategy.OCCURRENCE;
    }
    else if (value.equalsIgnoreCase("yield")) {
      return CoverageCountStrategy.YIELD;
    }
    else if (value.equalsIgnoreCase("product")) {
      return CoverageCountStrategy.PRODUCT;
    }
    else {
      throw new RuntimeException("Unknown index:" + value);
    }
  }

  @Override
  public String toString() {
    switch (this.index) {
    case 0:
      return "COUNT";
    case 1:
      return "LEVEL";
    case 2:
      return "OCCURRENCE";
    case 3:
      return "YIELD";
    case 4:
      return "PRODUCT";
    default:
      throw new RuntimeException("Unknown index");
    }
  }
}
