package net.seninp.grammarviz.view.table;

import net.seninp.jmotif.sax.discord.DiscordRecords;

/**
 * Table Data Model for the sequitur JTable
 * 
 * @author seninp
 * 
 */
public class AnomalyTableModel extends GrammarvizRulesTableDataModel {

  /** Fancy serial. */
  private static final long serialVersionUID = -2952232752352693293L;

  /**
   * Constructor.
   */
  public AnomalyTableModel() {
    AnomalyTableColumns[] columns = AnomalyTableColumns.values();
    String[] schemaColumns = new String[columns.length];
    for (int i = 0; i < columns.length; i++) {
      schemaColumns[i] = columns[i].getColumnName();
    }
    setSchema(schemaColumns);
  }

  public void update(DiscordRecords discords) {
    int rowIndex = 0;
    rows.clear();
    if (!(null == discords)) {
      for (rowIndex = 0; rowIndex < discords.getSize(); rowIndex++) {
        Object[] item = new Object[getColumnCount() + 1];
        int nColumn = 0;
        item[nColumn++] = rowIndex;
        item[nColumn++] = discords.get(rowIndex).getPosition();
        item[nColumn++] = discords.get(rowIndex).getLength();
        item[nColumn++] = Double.valueOf(discords.get(rowIndex).getNNDistance());
        item[nColumn++] = discords.get(rowIndex).getRuleId();
        rows.add(item);
      }
    }
    fireTableDataChanged();
  }

  /*
   * Important for table column sorting (non-Javadoc)
   * 
   * @see javax.swing.table.AbstractTableModel#getColumnClass(int)
   */
  public Class<?> getColumnClass(int columnIndex) {
    /*
     * for the RuleNumber and RuleFrequency column we use column class Integer.class so we can sort
     * it correctly in numerical order
     */
    if (columnIndex == AnomalyTableColumns.ANOMALY_RANK.ordinal())
      return Integer.class;
    if (columnIndex == AnomalyTableColumns.ANOMALY_POSITION.ordinal())
      return Integer.class;
    if (columnIndex == AnomalyTableColumns.ANOMALY_LENGTH.ordinal())
      return Integer.class;
    if (columnIndex == AnomalyTableColumns.ANOMALY_NNDISTANCE.ordinal())
      return Double.class;
    if (columnIndex == AnomalyTableColumns.ANOMALY_RULE.ordinal())
      return Integer.class;

    return String.class;
  }

}
