package net.seninp.grammarviz.view.table;

/**
 * Enum for the columns in the sequitur JTable.
 * 
 * @author Manfred Lerner, seninp
 * 
 */
public enum PrunedRulesTableColumns {
  
  CLASS_NUMBER("Class index"), 
  SUBSEQUENCE_NUMBER("Sub_Sequences number"),
  MIN_LENGTH("Minimal length"),
  MAX_LENGTH("Maximal length");

  private final String columnName;

  PrunedRulesTableColumns(String columnName) {
    this.columnName = columnName;
  }

  public String getColumnName() {
    return columnName;
  }

}
