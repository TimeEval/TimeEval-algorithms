package net.seninp.grammarviz.gi;

/**
 * GI algorithm selector.
 * 
 * @author psenin
 * 
 */
public enum GIAlgorithm {

  /** Sequitur. */
  SEQUITUR(0),

  /** Re-Pair. */
  REPAIR(1);

  private final int algIndex;

  /**
   * Constructor.
   * 
   * @param algIdx The algorithm index.
   */
  GIAlgorithm(int algIdx) {
    this.algIndex = algIdx;
  }

  /**
   * Get the alg index.
   * 
   * @return alg index.
   */
  public int toAlgIndex() {
    return this.algIndex;
  }

  /**
   * Parse the numerical value into an instance.
   * 
   * @param value the string value.
   * @return new instance.
   */
  public static GIAlgorithm fromValue(int value) {
    if (0 == value) {
      return GIAlgorithm.SEQUITUR;
    }
    else if (1 == value) {
      return GIAlgorithm.REPAIR;
    }
    else {
      throw new RuntimeException("Unknown index:" + value);
    }

  }

  /**
   * Parse the string value into an instance.
   * 
   * @param value the string value.
   * @return new instance.
   */
  public static GIAlgorithm fromValue(String value) {
    if ("sequitur".equalsIgnoreCase(value) || "s".equalsIgnoreCase(value)) {
      return GIAlgorithm.SEQUITUR;
    }
    else if ("repair".equalsIgnoreCase(value) || "re-pair".equalsIgnoreCase(value)
        || "r".equalsIgnoreCase(value)) {
      return GIAlgorithm.REPAIR;
    }
    else {
      throw new RuntimeException("Unknown index:" + value);
    }
  }

  /**
   * {@inheritDoc}
   */
  public String toString() {
    switch (this.algIndex) {
    case 0:
      return "Sequitur";
    case 1:
      return "Re-Pair";
    default:
      throw new RuntimeException("Unknown index:" + this.algIndex);
    }
  }
}
