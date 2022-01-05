package net.seninp.grammarviz.logic;

import java.util.ArrayList;

/**
 * A data structure which is a collection of motifs.
 * 
 * @author psenin
 */
public class SameLengthMotifs {

  private ArrayList<SAXMotif> sameLenMotifs;

  private int minMotifLen;
  private int maxMotifLen;

  /**
   * @return the sameLenMotifs
   */
  public ArrayList<SAXMotif> getSameLenMotifs() {
    return sameLenMotifs;
  }

  /**
   * @param sameLenMotifs the sameLenMotifs to set
   */
  public void setSameLenMotifs(ArrayList<SAXMotif> sameLenMotifs) {
    this.sameLenMotifs = sameLenMotifs;
  }

  /**
   * @return the minMotifLen
   */
  public int getMinMotifLen() {
    return minMotifLen;
  }

  /**
   * @param minMotifLen the minMotifLen to set
   */
  public void setMinMotifLen(int minMotifLen) {
    this.minMotifLen = minMotifLen;
  }

  /**
   * @return the maxMotifLen
   */
  public int getMaxMotifLen() {
    return maxMotifLen;
  }

  /**
   * @param maxMotifLen the maxMotifLen to set
   */
  public void setMaxMotifLen(int maxMotifLen) {
    this.maxMotifLen = maxMotifLen;
  }

}
