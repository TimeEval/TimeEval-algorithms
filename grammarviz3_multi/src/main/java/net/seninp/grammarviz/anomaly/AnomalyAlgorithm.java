package net.seninp.grammarviz.anomaly;

/**
 * The anomaly discovery algorithm selector.
 * 
 * @author psenin
 *
 */
public enum AnomalyAlgorithm {
  BRUTEFORCE(0), HOTSAX(1), RRA(2), RRAPRUNED(3), RRASAMPLED(4), EXPERIMENT(5);

  private final int index;

  AnomalyAlgorithm(int index) {
    this.index = index;
  }

  public int index() {
    return index;
  }

  public static AnomalyAlgorithm fromValue(int value) {
    switch (value) {
    case 0:
      return AnomalyAlgorithm.BRUTEFORCE;
    case 1:
      return AnomalyAlgorithm.HOTSAX;
    case 2:
      return AnomalyAlgorithm.RRA;
    case 3:
      return AnomalyAlgorithm.RRAPRUNED;
    case 4:
      return AnomalyAlgorithm.RRASAMPLED;
    case 5:
      return AnomalyAlgorithm.EXPERIMENT;
    default:
      throw new RuntimeException("Unknown index:" + value);
    }
  }

  public static AnomalyAlgorithm fromValue(String value) {
    if (value.equalsIgnoreCase("bruteforce")) {
      return AnomalyAlgorithm.BRUTEFORCE;
    }
    else if (value.equalsIgnoreCase("hotsaxtrie")) {
      return AnomalyAlgorithm.HOTSAX;
    }
    else if (value.equalsIgnoreCase("rra")) {
      return AnomalyAlgorithm.RRA;
    }
    else if (value.equalsIgnoreCase("rrapruned")) {
      return AnomalyAlgorithm.RRAPRUNED;
    }
    else if (value.equalsIgnoreCase("rrasampled")) {
      return AnomalyAlgorithm.RRASAMPLED;
    }
    else if (value.equalsIgnoreCase("experiment")) {
      return AnomalyAlgorithm.EXPERIMENT;
    }
    else {
      throw new RuntimeException("Unknown index:" + value);
    }
  }

  @Override
  public String toString() {
    switch (this.index) {
    case 0:
      return "BRUTEFORCE";
    case 1:
      return "HOTSAX";
    case 2:
      return "RRA";
    case 3:
      return "RRAPRUNED";
    case 4:
      return "RRASAMPLED";
    case 5:
      return "EXPERIMENT";
    default:
      throw new RuntimeException("Unknown index");
    }
  }
}
