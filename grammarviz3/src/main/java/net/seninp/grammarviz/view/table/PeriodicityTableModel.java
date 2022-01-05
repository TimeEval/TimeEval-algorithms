package net.seninp.grammarviz.view.table;

import net.seninp.gi.logic.GrammarRuleRecord;
import net.seninp.gi.logic.GrammarRules;

/**
 * Table Data Model for the sequitur JTable
 * 
 * @author seninp
 * 
 */
public class PeriodicityTableModel extends GrammarvizRulesTableDataModel {

  /** Fancy serial. */
  private static final long serialVersionUID = -2952232752352693293L;

  /**
   * Constructor.
   */
  public PeriodicityTableModel() {
    PeriodicityTableColumns[] columns = PeriodicityTableColumns.values();
    String[] schemaColumns = new String[columns.length];
    for (int i = 0; i < columns.length; i++) {
      schemaColumns[i] = columns[i].getColumnName();
    }
    setSchema(schemaColumns);
  }

  public void update(GrammarRules grammarRules) {
    rows.clear();
    if (!(null == grammarRules)) {
      // TODO: it breaks here assuming the sequential order of rules... see Issue #23
      //
      int ruleCounter = 0;
      while (ruleCounter < grammarRules.size()) {
        GrammarRuleRecord rule = grammarRules.get(ruleCounter);
        if (null != rule) {
          Object[] item = new Object[getColumnCount() + 1];
          int nColumn = 0;
          item[nColumn++] = rule.ruleNumber();
          item[nColumn++] = rule.getOccurrences().size();
          item[nColumn++] = rule.getMeanLength();
          item[nColumn++] = rule.getPeriod();
          item[nColumn++] = rule.getPeriodError();
          rows.add(item);
        }
        ruleCounter++;
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
    if (columnIndex == PeriodicityTableColumns.RULE_NUMBER.ordinal())
      return Integer.class;
    if (columnIndex == PeriodicityTableColumns.RULE_FREQUENCY.ordinal())
      return Integer.class;
    if (columnIndex == PeriodicityTableColumns.LENGTH.ordinal())
      return Integer.class;
    if (columnIndex == PeriodicityTableColumns.PERIOD.ordinal())
      return Double.class;
    if (columnIndex == PeriodicityTableColumns.PERIOD_ERROR.ordinal())
      return Double.class;

    return String.class;
  }

}
