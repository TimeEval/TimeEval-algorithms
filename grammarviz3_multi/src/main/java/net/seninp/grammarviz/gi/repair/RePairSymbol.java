package net.seninp.grammarviz.gi.repair;

import java.util.Arrays;

/**
 * The symbol -- which essentially is a token.
 * 
 * @author psenin
 * 
 */
public class RePairSymbol {

  /**
   * Payload.
   */
  private char[] string;

  /**
   * Position of the symbol in the string.
   */
  private Integer stringPosition;

  /**
   * Constructor.
   */
  public RePairSymbol() {
    super();
    this.stringPosition = null;
  }

  /**
   * Constructor.
   * 
   * @param token the payload.
   * @param stringPosition the position of the symbol in the string.
   */
  public RePairSymbol(String token, int stringPosition) {
    super();
    this.string = token.toCharArray();
    this.stringPosition = stringPosition;
  }

  /**
   * This is overridden in Guard.
   * 
   * @return true if the symbol is the guard.
   */
  public boolean isGuard() {
    return false;
  }

  /**
   * The position getter.
   * 
   * @return The symbol position in the string.
   */
  public int getStringPosition() {
    return this.stringPosition;
  }

  /**
   * The position setter.
   * 
   * @param saxStringPosition the position to set.
   */
  public void setStringPosition(int saxStringPosition) {
    this.stringPosition = saxStringPosition;
  }

  /**
   * This will be overridden in the non-Terminal symbol, i.e. guard.
   * 
   * @return The rule hierarchy level.
   */
  public int getLevel() {
    return 0;
  }
  
  public String toExpandedString() {
    return String.valueOf(this.string);
  }

  public String toString() {
    return String.valueOf(this.string);
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + Arrays.hashCode(string);
    result = prime * result + ((stringPosition == null) ? 0 : stringPosition.hashCode());
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
    RePairSymbol other = (RePairSymbol) obj;
    if (!Arrays.equals(string, other.string))
      return false;
    if (stringPosition == null) {
      if (other.stringPosition != null)
        return false;
    }
    else if (!stringPosition.equals(other.stringPosition))
      return false;
    return true;
  }

}
