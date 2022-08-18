package net.seninp.grammarviz.gi.repair;

import net.seninp.grammarviz.gi.repair.NewRepair;
import net.seninp.jmotif.sax.datastructure.SAXRecords;

/**
 * Implements RePair.
 * 
 * @author psenin
 * 
 */
public final class RePairFactory {

  private static final String SPACE = " ";

  // the logger
  //
  // private static final Logger LOGGER = LoggerFactory.getLogger(RePairFactory.class);

  /**
   * Disable constructor.
   */
  private RePairFactory() {
    assert true;
  }

  /**
   * Builds a repair grammar given a set of SAX records.
   * 
   * @param saxRecords the records to process.
   * 
   * @return the grammar.
   */
  public static RePairGrammar buildGrammar(SAXRecords saxRecords) {

    RePairGrammar grammar = NewRepair.parse(saxRecords.getSAXString(SPACE));

    return grammar;

  }

  /**
   * Builds a grammar given a string of terminals delimeted by space.
   * 
   * @param inputString the input string.
   * @return the RePair grammar.
   */
  public static RePairGrammar buildGrammar(String inputString) {

    RePairGrammar grammar = NewRepair.parse(inputString);

    return grammar;

  }

}
