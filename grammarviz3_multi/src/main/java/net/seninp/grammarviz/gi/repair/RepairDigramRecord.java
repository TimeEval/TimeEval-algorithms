package net.seninp.grammarviz.gi.repair;

/**
 * The RePair priority queue entry.
 * 
 * @author psenin
 *
 */
public class RepairDigramRecord implements Comparable<RepairDigramRecord> {

  // the digram
  protected String str;

  // and its observed frequency
  protected int freq;

  /**
   * Constructor.
   * 
   * @param key the digram string.
   * @param frequency digram's frequency.
   */
  public RepairDigramRecord(String key, int frequency) {
    this.str = key;
    this.freq = frequency;
  }

  /**
   * A comparator built upon occurrence frequency only.
   */
  @Override
  public int compareTo(RepairDigramRecord o) {
    if (this.freq > o.freq) {
      return 1;
    }
    else if (this.freq < o.freq) {
      return -1;
    }
    return 0;
  }

  /**
   * Get the digram string.
   * 
   * @return the digram string.
   */
  public String getDigram() {
    return this.str;
  }

  /**
   * The frequency getter.
   * 
   * @return the frequency.
   */
  public int getFrequency() {
    return this.freq;
  }

  public String toString() {
    return this.str + ":" + this.freq;
  };

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + freq;
    result = prime * result + ((str == null) ? 0 : str.hashCode());
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
    RepairDigramRecord other = (RepairDigramRecord) obj;
    if (freq != other.freq)
      return false;
    if (str == null) {
      if (other.str != null)
        return false;
    }
    else if (!str.equals(other.str))
      return false;
    return true;
  }

}
