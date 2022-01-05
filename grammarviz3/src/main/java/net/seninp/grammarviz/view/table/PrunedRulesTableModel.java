package net.seninp.grammarviz.view.table;

import java.util.ArrayList;
import net.seninp.grammarviz.logic.PackedRuleRecord;

/**
 * Table Data Model for the reduced and packed rules JTable
 * 
 * @author seninp
 * 
 */
public class PrunedRulesTableModel extends PrunedRulesTableDataModel {

  /** Fancy serial. */
  private static final long serialVersionUID = -2952232752352963293L;

  /**
   * Constructor.
   */
  public PrunedRulesTableModel() {
	  PrunedRulesTableColumns[] columns = PrunedRulesTableColumns.values();
    String[] schemaColumns = new String[columns.length];
    for (int i = 0; i < columns.length; i++) {
      schemaColumns[i] = columns[i].getColumnName();
    }
    setSchema(schemaColumns);
  }

  public void update(ArrayList<PackedRuleRecord> packedRulesSet) {
    int rowIndex = 0;
    rows.clear();
    for (rowIndex = 0; rowIndex < packedRulesSet.size(); rowIndex++) {

      Object[] item = new Object[getColumnCount() + 1];

      int nColumn = 0;

      item[nColumn++] = packedRulesSet.get(rowIndex).getClassIndex();
      item[nColumn++] = packedRulesSet.get(rowIndex).getSubsequenceNumber();
      item[nColumn++] = packedRulesSet.get(rowIndex).getMinLength();
      item[nColumn++] = packedRulesSet.get(rowIndex).getMaxLength();

      rows.add(item);
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
    if (columnIndex == PrunedRulesTableColumns.CLASS_NUMBER.ordinal())
      return Integer.class;
    if (columnIndex == PrunedRulesTableColumns.SUBSEQUENCE_NUMBER.ordinal())
      return Integer.class;
    if (columnIndex == PrunedRulesTableColumns.MIN_LENGTH.ordinal())
      return Integer.class;
    if (columnIndex == PrunedRulesTableColumns.MAX_LENGTH.ordinal())
      return Integer.class;
    
    return Integer.class;
  }

}
