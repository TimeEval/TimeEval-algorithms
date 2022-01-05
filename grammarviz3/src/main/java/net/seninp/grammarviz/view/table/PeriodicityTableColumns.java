package net.seninp.grammarviz.view.table;

/**
 * Enum for the columns in Sequitur JTable corresponding to Motifs.
 * 
 * @author seninp
 * 
 */
public enum PeriodicityTableColumns {
  
  // set of enumerated rules
  //
  RULE_NUMBER("R#"), 
  RULE_FREQUENCY("Frequency"),
  LENGTH("Length"),
  PERIOD("Period"),
  PERIOD_ERROR("Period error");

  private final String columnName;

  PeriodicityTableColumns(String columnName) {
    this.columnName = columnName;
  }

  public String getColumnName() {
    return columnName;
  }

}
