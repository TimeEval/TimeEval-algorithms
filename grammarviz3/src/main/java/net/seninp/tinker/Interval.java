package net.seninp.tinker;

/**
 * Implements an interval. Start inclusive, end exclusive.
 * 
 * @author psenin
 * 
 */
public class Interval {

  private int start;
  private int end;
  private double coverage;

  /**
   * Constructor; start inclusive, end exclusive.
   * 
   * @param start the interval's start.
   * @param end the interval's end.
   */
  public Interval(int start, int end) {
    this.start = start;
    this.end = end;
    this.coverage = -1;
  }

  /**
   * Constructor; start inclusive, end exclusive.
   * 
   * @param start the interval's start.
   * @param end the interval's end.
   * @param coverage the interval's coverage.
   */
  public Interval(int start, int end, double coverage) {
    this.start = start;
    this.end = end;
    this.coverage = coverage;
  }

  public double getCoverage() {
    return coverage;
  }

  public void setCoverage(double coverage) {
    this.coverage = coverage;
  }

  public void setStart(int start) {
    this.start = start;
  }

  public int getStart() {
    return this.start;
  }

  public void setEnd(int end) {
    this.end = end;
  }

  public int getEnd() {
    return this.end;
  }

  public int getLength() {
    return Math.abs(this.end - this.start);
  }

  /**
   * Returns true if this interval intersects the specified interval.
   *
   * @param that the other interval
   * @return <tt>true</tt> if this interval intersects the argument interval; <tt>false</tt>
   * otherwise
   */
  public boolean intersects(Interval that) {
    if (this.end < that.start) {
      return false;
    }
    if (that.end < this.start) {
      return false;
    }
    return true;
  }

  /**
   * Returns true if this interval contains the specified value.
   *
   * @param x the value
   * @return <tt>true</tt> if this interval contains the value <tt>x</tt>; <tt>false</tt> otherwise
   */
  public boolean contains(int x) {
    return (start <= x) && (x <= end);
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    long temp;
    temp = Double.doubleToLongBits(coverage);
    result = prime * result + (int) (temp ^ (temp >>> 32));
    result = prime * result + end;
    result = prime * result + start;
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
    Interval other = (Interval) obj;
    if (Double.doubleToLongBits(coverage) != Double.doubleToLongBits(other.coverage))
      return false;
    if (end != other.end)
      return false;
    if (start != other.start)
      return false;
    return true;
  }

}
