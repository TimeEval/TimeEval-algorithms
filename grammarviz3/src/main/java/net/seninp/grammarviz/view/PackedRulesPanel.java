package net.seninp.grammarviz.view;

import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.util.ArrayList;
import java.util.Arrays;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.ListSelectionModel;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;
import javax.swing.table.JTableHeader;
import javax.swing.table.TableRowSorter;
import org.jdesktop.swingx.JXTable;
import org.jdesktop.swingx.JXTableHeader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.seninp.grammarviz.logic.GrammarVizChartData;
import net.seninp.grammarviz.view.table.GrammarvizRulesTableColumns;
import net.seninp.grammarviz.view.table.PrunedRulesTableModel;

/**
 * 
 * handling the chart panel and sequitur rules table
 * 
 * 
 */

public class PackedRulesPanel extends JPanel
    implements ListSelectionListener, PropertyChangeListener {

  /** Fancy serial. */
  private static final long serialVersionUID = -2710973854572981568L;

  public static final String FIRING_PROPERTY_PACKED = "selectedRow_packed";

  private PrunedRulesTableModel packedTableModel = new PrunedRulesTableModel();

  private JXTable packedTable;

  private GrammarVizChartData chartData;

  private JScrollPane packedRulesPane;

  // private String selectedRule;

  private ArrayList<String> selectedRules;

  private boolean acceptListEvents;

  // static block - we instantiate the logger
  //
  private static final Logger LOGGER = LoggerFactory.getLogger(PackedRulesPanel.class);

  // private Comparator<String> expandedRuleComparator = new Comparator<String>() {
  // public int compare(String s1, String s2) {
  // return s1.length() - s2.length();
  // }
  // };

  /**
   * Constructor.
   */
  public PackedRulesPanel() {
    super();
    this.packedTableModel = new PrunedRulesTableModel();
    this.packedTable = new JXTable() {

      private static final long serialVersionUID = 2L;

      @Override
      protected JTableHeader createDefaultTableHeader() {
        return new JXTableHeader(columnModel) {
          private static final long serialVersionUID = 1L;

          @Override
          public void updateUI() {
            super.updateUI();
            // need to do in updateUI to survive toggling of LAF
            if (getDefaultRenderer() instanceof JLabel) {
              ((JLabel) getDefaultRenderer()).setHorizontalAlignment(JLabel.CENTER);

            }
          }
        };
      }

    };
    this.packedTable.setModel(packedTableModel);
    this.packedTable.getSelectionModel().setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
    this.packedTable.setShowGrid(false);

    this.packedTable.getSelectionModel().addListSelectionListener(this);

    @SuppressWarnings("unused")
    org.jdesktop.swingx.renderer.DefaultTableRenderer renderer = (org.jdesktop.swingx.renderer.DefaultTableRenderer) packedTable
        .getDefaultRenderer(String.class);

    TableRowSorter<PrunedRulesTableModel> sorter = new TableRowSorter<PrunedRulesTableModel>(
        packedTableModel);
    packedTable.setRowSorter(sorter);
    // sorter.setComparator(PackedTableColumns.CLASS_NUMBER.ordinal(),
    // expandedRuleComparator);

    this.packedRulesPane = new JScrollPane(packedTable);
  }

  /**
   * Set the new data.
   * 
   * @param chartData the new data.
   */
  public void setChartData(GrammarVizChartData chartData) {

    this.acceptListEvents = false;

    // save the data
    this.chartData = chartData;

    // update
    packedTableModel.update(this.chartData.getArrPackedRuleRecords());

    // put new data on show
    resetPanel();

    this.acceptListEvents = true;
  }

  /**
   * create the panel with the sequitur rules table
   * 
   * @return sequitur panel
   */
  public void resetPanel() {
    // cleanup all the content
    this.removeAll();
    this.add(packedRulesPane);
    this.validate();
    this.repaint();
  }

  /**
   * @return packed table model
   */
  public PrunedRulesTableModel getPackedTableModel() {
    return packedTableModel;
  }

  /**
   * @return sequitur table
   */
  public JTable getSequiturTable() {
    return packedTable;
  }

  @Override
  public void valueChanged(ListSelectionEvent arg) {

    if (!arg.getValueIsAdjusting() && this.acceptListEvents) {
      int[] rows = packedTable.getSelectedRows();
      LOGGER.debug("Selected ROWS: " + Arrays.toString(rows));
      ArrayList<String> rules = new ArrayList<String>(rows.length);
      for (int i = 0; i < rows.length; i++) {
        int ridx = rows[i];
        String rule = String.valueOf(
            packedTable.getValueAt(ridx, GrammarvizRulesTableColumns.RULE_NUMBER.ordinal()));
        rules.add(rule);
      }
      this.firePropertyChange(FIRING_PROPERTY_PACKED, this.selectedRules, rules);
      this.selectedRules = rules;
    }

    // if (!arg.getValueIsAdjusting() && this.acceptListEvents) {
    // int col = packedTable.getSelectedColumn();
    // int row = packedTable.getSelectedRow();
    // consoleLogger.debug("Selected ROW: " + row + " - COL: " + col);
    // String rule = String.valueOf(packedTable.getValueAt(row,
    // PrunedRulesTableColumns.CLASS_NUMBER.ordinal()));
    // this.firePropertyChange(FIRING_PROPERTY_PACKED, this.selectedRule, rule);
    // this.selectedRule = rule;
    // }
  }

  /**
   * Resets the selection and resorts the table by the Rules.
   */
  public void resetSelection() {
    // TODO: there is the bug. commented out.
    // sequiturTable.getSelectionModel().clearSelection();
    // sequiturTable.setSortOrder(0, SortOrder.ASCENDING);
  }

  @Override
  public void propertyChange(PropertyChangeEvent arg0) {
    // TODO Auto-generated method stub

  }

  // public void propertyChange(PropertyChangeEvent event) {
  // String prop = event.getPropertyName();
  //
  // if (prop.equalsIgnoreCase(SequiturMessage.MAIN_CHART_CLICKED_MESSAGE)) {
  // String rule = (String) event.getNewValue();
  // for (int row = 0; row <= sequiturTable.getRowCount() - 1; row++) {
  // for (int col = 0; col <= sequiturTable.getColumnCount() - 1; col++) {
  // if (rule.equals(chartData.convert2OriginalSAXAlphabet('1',
  // sequiturTable.getValueAt(row, col).toString()))) {
  // sequiturTable.scrollRectToVisible(sequiturTable.getCellRect(row, 0, true));
  // sequiturTable.setRowSelectionInterval(row, row);
  // }
  // }
  // }
  // }
  // }

}
