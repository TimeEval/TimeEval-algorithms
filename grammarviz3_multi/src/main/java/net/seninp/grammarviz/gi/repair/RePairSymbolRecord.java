package net.seninp.grammarviz.gi.repair;

import net.seninp.grammarviz.gi.repair.RePairSymbol;

/**
 * Used for keeping the track of repair symbols in the string.
 * 
 * @author psenin
 *
 */
public class RePairSymbolRecord {

  /** The actual symbol. */
  private RePairSymbol payload;

  /** Since it's a doubly-linked list, we keep track of before and after symbols. */
  private RePairSymbolRecord next;
  private RePairSymbolRecord prev;

  /**
   * Constructor.
   * 
   * @param symbol the symbol to wrap -- the payload.
   */
  public RePairSymbolRecord(RePairSymbol symbol) {
    this.payload = symbol;
  }

  /**
   * The payload getter.
   * 
   * @return the wrapped symbol.
   */
  public RePairSymbol getPayload() {
    return payload;
  }

  /**
   * The next symbol setter.
   * 
   * @param sr the next symbol pointer.
   */
  public void setNext(RePairSymbolRecord sr) {
    this.next = sr;
  }

  /**
   * The prev symbol setter.
   * 
   * @param sr the prev symbol pointer.
   */
  public void setPrevious(RePairSymbolRecord sr) {
    this.prev = sr;
  }

  /**
   * Next symbol getter.
   * 
   * @return next symbol.
   */
  public RePairSymbolRecord getNext() {
    return this.next;
  }

  /**
   * Previous symbol getter.
   * 
   * @return previous symbol.
   */
  public RePairSymbolRecord getPrevious() {
    return this.prev;
  }

  /**
   * An index getter, calls the payload's method for that.
   * 
   * @return the index in the string, or -1 if the payload is NULL.
   */
  public int getIndex() {
    if (null == this.payload) {
      return -1;
    }
    return this.payload.getStringPosition();
  }

  /**
   * {@inheritDoc}
   */
  public String toString() {
    if (null == this.payload) {
      return "null";
    }
    return this.payload.toString();
  }
}
