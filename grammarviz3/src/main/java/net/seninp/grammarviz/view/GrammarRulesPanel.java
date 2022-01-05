package net.seninp.grammarviz.view;

import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.ListSelectionModel;
import javax.swing.SwingConstants;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.JTableHeader;
import javax.swing.table.TableColumnModel;
import javax.swing.table.TableRowSorter;
import org.jdesktop.swingx.JXTable;
import org.jdesktop.swingx.JXTableHeader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.seninp.grammarviz.model.GrammarVizMessage;
import net.seninp.grammarviz.session.UserSession;
import net.seninp.grammarviz.view.table.GrammarvizRulesTableColumns;
import net.seninp.grammarviz.view.table.GrammarvizRulesTableModel;

/**
 * 
 * Implements the rules panel.
 * 
 * @author Manfred Lerner, seninp
 * 
 */

public class GrammarRulesPanel extends JPanel
    implements ListSelectionListener, PropertyChangeListener {

  /** Fancy serial. */
  private static final long serialVersionUID = -2710973854572981568L;

  public static final String FIRING_PROPERTY = "selectedRows";

  private GrammarvizRulesTableModel sequiturTableModel = new GrammarvizRulesTableModel();

  private JXTable sequiturTable;

  private UserSession session;

  private JScrollPane sequiturRulesPane;

  private ArrayList<String> selectedRules;

  private boolean acceptListEvents;

  // static block - we instantiate the logger
  //
  private static final Logger LOGGER = LoggerFactory.getLogger(GrammarRulesPanel.class);

  /*
   * 
   * Comparator for the sorting of the Expanded Sequitur Rules Easy logic: sort by the length of the
   * Expanded Sequitur Rules
   */
  private Comparator<String> expandedRuleComparator = new Comparator<String>() {
    public int compare(String s1, String s2) {
      return s1.length() - s2.length();
    }
  };

  /**
   * Constructor.
   */
  public GrammarRulesPanel() {
    super();
    this.sequiturTableModel = new GrammarvizRulesTableModel();
    this.sequiturTable = new JXTable() {

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

    this.sequiturTable.setModel(sequiturTableModel);
    // this.sequiturTable.getSelectionModel().setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
    this.sequiturTable.getSelectionModel()
        .setSelectionMode(ListSelectionModel.MULTIPLE_INTERVAL_SELECTION);
    this.sequiturTable.setShowGrid(false);

    this.sequiturTable.getSelectionModel().addListSelectionListener(this);

    @SuppressWarnings("unused")
    org.jdesktop.swingx.renderer.DefaultTableRenderer renderer = (org.jdesktop.swingx.renderer.DefaultTableRenderer) sequiturTable
        .getDefaultRenderer(String.class);

    // Make some columns wider than the rest, so that the info fits in.
    TableColumnModel columnModel = sequiturTable.getColumnModel();
    columnModel.getColumn(GrammarvizRulesTableColumns.RULE_NUMBER.ordinal()).setPreferredWidth(30);
    columnModel.getColumn(GrammarvizRulesTableColumns.RULE_USE_FREQUENCY.ordinal())
        .setPreferredWidth(40);
    columnModel.getColumn(GrammarvizRulesTableColumns.SEQUITUR_RULE.ordinal())
        .setPreferredWidth(100);
    columnModel.getColumn(GrammarvizRulesTableColumns.EXPANDED_SEQUITUR_RULE.ordinal())
        .setPreferredWidth(150);
    columnModel.getColumn(GrammarvizRulesTableColumns.RULE_MEAN_LENGTH.ordinal())
        .setPreferredWidth(120);

    TableRowSorter<GrammarvizRulesTableModel> sorter = new TableRowSorter<GrammarvizRulesTableModel>(
        sequiturTableModel);
    sequiturTable.setRowSorter(sorter);
    sorter.setComparator(GrammarvizRulesTableColumns.EXPANDED_SEQUITUR_RULE.ordinal(),
        expandedRuleComparator);

    DefaultTableCellRenderer rightRenderer = new DefaultTableCellRenderer();
    rightRenderer.setHorizontalAlignment(SwingConstants.RIGHT);
    this.sequiturTable.getColumnModel().getColumn(5).setCellRenderer(rightRenderer);

    this.sequiturRulesPane = new JScrollPane(sequiturTable);
  }

  /**
   * create the panel with the sequitur rules table
   * 
   * @return sequitur panel
   */
  public void resetPanel() {
    // cleanup all the content
    this.removeAll();
    this.add(sequiturRulesPane);
    this.acceptListEvents = false;
    sequiturTableModel.update(this.session.chartData.getGrammarRules());
    this.acceptListEvents = true;
    this.revalidate();
    this.repaint();
  }

  /**
   * @return sequitur table model
   */
  public GrammarvizRulesTableModel getSequiturTableModel() {
    return sequiturTableModel;
  }

  /**
   * @return sequitur table
   */
  public JTable getSequiturTable() {
    return sequiturTable;
  }

  @Override
  public void valueChanged(ListSelectionEvent arg) {

    if (!arg.getValueIsAdjusting() && this.acceptListEvents) {
      int[] rows = sequiturTable.getSelectedRows();
      LOGGER.debug("Selected ROWS: " + Arrays.toString(rows));
      ArrayList<String> rules = new ArrayList<String>(rows.length);
      for (int i = 0; i < rows.length; i++) {
        int ridx = rows[i];
        String rule = String.valueOf(
            sequiturTable.getValueAt(ridx, GrammarvizRulesTableColumns.RULE_NUMBER.ordinal()));
        rules.add(rule);
      }
      this.firePropertyChange(FIRING_PROPERTY, this.selectedRules, rules);
      this.selectedRules = rules;
    }
  }

  /**
   * Resets the selection and resorts the table by the Rules.
   */
  public void resetSelection() {
    // TODO: there is the bug. commented out.
    sequiturTable.getSelectionModel().clearSelection();
    // sequiturTable.setSortOrder(0, SortOrder.ASCENDING);
  }

  public void propertyChange(PropertyChangeEvent event) {
    String prop = event.getPropertyName();

    if (prop.equalsIgnoreCase(GrammarVizMessage.MAIN_CHART_CLICKED_MESSAGE)) {
      String rule = (String) event.getNewValue();
      for (int row = 0; row <= sequiturTable.getRowCount() - 1; row++) {
        for (int col = 0; col <= sequiturTable.getColumnCount() - 1; col++) {
          if (rule.equals(this.session.chartData.convert2OriginalSAXAlphabet('1',
              sequiturTable.getValueAt(row, col).toString()))) {
            sequiturTable.scrollRectToVisible(sequiturTable.getCellRect(row, 0, true));
            sequiturTable.setRowSelectionInterval(row, row);
          }
        }
      }
    }
  }

  /**
   * Clears the panel.
   */
  public void clearPanel() {
    this.acceptListEvents = false;
    this.removeAll();
    sequiturTableModel.update(null);
    this.validate();
    this.repaint();
    this.acceptListEvents = true;
  }

  public void setChartData(UserSession session) {
    clearPanel();
    this.session = session;
    resetPanel();
  }

}
