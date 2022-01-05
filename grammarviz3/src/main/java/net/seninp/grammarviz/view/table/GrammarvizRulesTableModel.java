package net.seninp.grammarviz.view.table;

import net.seninp.gi.logic.GrammarRuleRecord;
import net.seninp.gi.logic.GrammarRules;

/**
 * Table Data Model for the sequitur JTable
 * 
 * @author Manfred Lerner, seninp
 * 
 */
public class GrammarvizRulesTableModel extends GrammarvizRulesTableDataModel {

  /** Fancy serial. */
  private static final long serialVersionUID = -2952232752352963293L;

  /**
   * Constructor.
   */
  public GrammarvizRulesTableModel() {
    GrammarvizRulesTableColumns[] columns = GrammarvizRulesTableColumns.values();
    String[] schemaColumns = new String[columns.length];
    for (int i = 0; i < columns.length; i++) {
      schemaColumns[i] = columns[i].getColumnName();
    }
    setSchema(schemaColumns);
  }

  /**
   * Updates the table model with provided data.
   * 
   * @param grammarRules the data for table.
   */
  public void update(GrammarRules grammarRules) {
    rows.clear();
    if (!(null == grammarRules)) {
      // TODO: it breaks here assuming the sequential order of rules... see Issue #23
      //
      int ruleCounter = 0;
      int rulesInTable = 0;
      while (rulesInTable < grammarRules.size()) {
        GrammarRuleRecord rule = grammarRules.get(ruleCounter);
        if (null != rule) {
          Object[] item = new Object[getColumnCount() + 1];
          int nColumn = 0;
          item[nColumn++] = rule.ruleNumber();
          item[nColumn++] = rule.getRuleLevel();
          item[nColumn++] = rule.getOccurrences().size();
          item[nColumn++] = rule.getRuleString();
          item[nColumn++] = rule.getExpandedRuleString();
          item[nColumn++] = rule.getRuleUseFrequency();
          item[nColumn++] = rule.getMeanLength();
          item[nColumn++] = rule.minMaxLengthAsString();
          // item[nColumn++] = saxContainerList.get(rowIndex).getOccurenceIndexes();
          rows.add(item);
          rulesInTable++;
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
    if (columnIndex == GrammarvizRulesTableColumns.RULE_NUMBER.ordinal())
      return Integer.class;
    if (columnIndex == GrammarvizRulesTableColumns.RULE_LEVEL.ordinal())
      return Integer.class;
    if (columnIndex == GrammarvizRulesTableColumns.RULE_FREQUENCY.ordinal())
      return Integer.class;
    if (columnIndex == GrammarvizRulesTableColumns.SEQUITUR_RULE.ordinal())
      return String.class;
    if (columnIndex == GrammarvizRulesTableColumns.EXPANDED_SEQUITUR_RULE.ordinal())
      return String.class;
    if (columnIndex == GrammarvizRulesTableColumns.RULE_USE_FREQUENCY.ordinal())
      return Integer.class;
    if (columnIndex == GrammarvizRulesTableColumns.RULE_MEAN_LENGTH.ordinal())
      return Integer.class;
    if (columnIndex == GrammarvizRulesTableColumns.LENGTH.ordinal())
      return String.class;
    // if (columnIndex == SequiturTableColumns.RULE_INDEXES.ordinal())
    // return String.class;

    return String.class;
  }

}
