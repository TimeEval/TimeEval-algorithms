package net.seninp.grammarviz.view.table;

/**
 * Enum for the columns in Sequitur JTable corresponding to Motifs.
 * 
 * @author seninp
 * 
 */
public enum AnomalyTableColumns {
  
  // set of enumerated rules
  //
  ANOMALY_RANK("Rank"), 
  ANOMALY_POSITION("Position"),
  ANOMALY_LENGTH("Length"),
  ANOMALY_NNDISTANCE("NN Distance"),
  ANOMALY_RULE("Grammar Rule");

  private final String columnName;

  AnomalyTableColumns(String columnName) {
    this.columnName = columnName;
  }

  public String getColumnName() {
    return columnName;
  }

}
