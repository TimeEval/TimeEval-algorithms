package net.seninp.grammarviz.view;

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
import net.seninp.grammarviz.session.UserSession;
import net.seninp.grammarviz.view.table.AnomalyTableModel;
import net.seninp.grammarviz.view.table.CellDoubleRenderer;
import net.seninp.grammarviz.view.table.GrammarvizRulesTableColumns;

public class GrammarVizAnomaliesPanel extends JPanel implements ListSelectionListener {

  /** Fancy serial. */
  private static final long serialVersionUID = -2710973845672981568L;

  public static final String FIRING_PROPERTY_ANOMALY = "selectedRow_anomaly";

  private UserSession session;

  private AnomalyTableModel anomalyTableModel;

  private JXTable anomalyTable;

  private JScrollPane anomaliesPane;

  private ArrayList<String> selectedAnomalies;

  private boolean acceptListEvents;

  // static block - we instantiate the logger
  //
  private static final Logger LOGGER = LoggerFactory.getLogger(GrammarRulesPanel.class);

  /**
   * Constructor.
   */
  public GrammarVizAnomaliesPanel() {
    super();
    this.anomalyTableModel = new AnomalyTableModel();
    this.anomalyTable = new JXTable() {

      private static final long serialVersionUID = 3L;

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
    this.anomalyTable.setModel(anomalyTableModel);
    // this.anomalyTable.getSelectionModel().setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
    this.anomalyTable.getSelectionModel()
        .setSelectionMode(ListSelectionModel.MULTIPLE_INTERVAL_SELECTION);
    this.anomalyTable.setShowGrid(false);
    this.anomalyTable.setDefaultRenderer(Double.class, new CellDoubleRenderer());

    this.anomalyTable.getSelectionModel().addListSelectionListener(this);

    @SuppressWarnings("unused")
    org.jdesktop.swingx.renderer.DefaultTableRenderer renderer = (org.jdesktop.swingx.renderer.DefaultTableRenderer) anomalyTable
        .getDefaultRenderer(String.class);

    TableRowSorter<AnomalyTableModel> sorter = new TableRowSorter<AnomalyTableModel>(
        anomalyTableModel);
    anomalyTable.setRowSorter(sorter);
    // sorter.setComparator(PackedTableColumns.CLASS_NUMBER.ordinal(),
    // expandedRuleComparator);

    this.anomaliesPane = new JScrollPane(anomalyTable);
  }

  /**
   * create the panel with the sequitur rules table
   * 
   * @return sequitur panel
   */
  public void resetPanel() {
    // cleanup all the content
    this.removeAll();
    this.add(anomaliesPane);
    this.validate();
    this.repaint();
  }

  /**
   * @return packed table model
   */
  public AnomalyTableModel getPeriodicityTableModel() {
    return anomalyTableModel;
  }

  /**
   * @return sequitur table
   */
  public JTable getAnomalyTable() {
    return anomalyTable;
  }

  @Override
  public void valueChanged(ListSelectionEvent arg) {
    if (!arg.getValueIsAdjusting() && this.acceptListEvents) {
      int[] rows = anomalyTable.getSelectedRows();
      LOGGER.debug("Selected ROWS: " + Arrays.toString(rows));
      ArrayList<String> rules = new ArrayList<String>(rows.length);
      for (int i = 0; i < rows.length; i++) {
        int ridx = rows[i];
        String rule = String.valueOf(
            anomalyTable.getValueAt(ridx, GrammarvizRulesTableColumns.RULE_NUMBER.ordinal()));
        rules.add(rule);
      }
      this.firePropertyChange(FIRING_PROPERTY_ANOMALY, this.selectedAnomalies, rules);
    }

  }

  public void updateAnomalies() {
    this.acceptListEvents = false;
    anomalyTableModel.update(this.session.chartData.getAnomalies());
    this.acceptListEvents = true;
  }

  /**
   * Clears the panel.
   */
  public void clear() {
    this.acceptListEvents = false;
    this.removeAll();
    this.session = null;
    anomalyTableModel.update(null);
    this.validate();
    this.repaint();
    this.acceptListEvents = true;
  }

  public void setChartData(UserSession session) {
    this.session = session;
    updateAnomalies();
  }

}
