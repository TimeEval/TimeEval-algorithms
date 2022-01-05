package net.seninp.grammarviz.view.table;

/**
 * Enum for the columns in Sequitur rules JTable.
 * 
 * @author Manfred Lerner, seninp
 * 
 */
public enum GrammarvizRulesTableColumns {
  
  RULE_NUMBER("R#"),
  RULE_LEVEL("Level"), 
  RULE_FREQUENCY("Frequency in R0"), 
  SEQUITUR_RULE("Rule"), 
  EXPANDED_SEQUITUR_RULE("Expanded Rule"),
  RULE_USE_FREQUENCY("Used"),
  RULE_MEAN_LENGTH("Mean length"),
  LENGTH("Min-max length");
  // RULE_INDEXES("String positions");

  private final String columnName;

  GrammarvizRulesTableColumns(String columnName) {
    this.columnName = columnName;
  }

  public String getColumnName() {
    return columnName;
  }

}
