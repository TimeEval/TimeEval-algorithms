package net.seninp.grammarviz.view;

import java.awt.Dimension;
import java.awt.Point;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JPanel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.miginfocom.swing.MigLayout;
import net.seninp.grammarviz.session.UserSession;

/* 1.4 example used by DialogDemo.java. */
class GrammarvizOptionsDialog extends JDialog implements ActionListener {

  private static final long serialVersionUID = -8273240774350932580L;

  private static final String OK_BUTTON_TEXT = "Save";
  private static final String CANCEL_BUTTON_TEXT = "Cancel";

  private UserSession session;
  private GrammarvizOptionsPane optionPane;

  // static block - we instantiate the logger
  //
  private static final Logger LOGGER = LoggerFactory.getLogger(GrammarvizOptionsDialog.class);

  /** Creates the reusable dialog. */
  public GrammarvizOptionsDialog(JFrame parentFrame, JPanel optionPanel, UserSession session) {

    super(parentFrame, true);

    if (parentFrame != null) {
      Dimension parentSize = parentFrame.getSize();
      Point p = parentFrame.getLocation();
      setLocation(p.x + parentSize.width / 4, p.y + parentSize.height / 4);
    }

    this.session = session;

    this.optionPane = (GrammarvizOptionsPane) optionPanel;

    MigLayout mainFrameLayout = new MigLayout("fill", "[grow,center]", "[grow]5[]");

    getContentPane().setLayout(mainFrameLayout);

    getContentPane().add(this.optionPane, "h 200:300:,w 500:550:,growx,growy,wrap");

    JPanel buttonPane = new JPanel();
    JButton okButton = new JButton(OK_BUTTON_TEXT);
    JButton cancelButton = new JButton(CANCEL_BUTTON_TEXT);
    buttonPane.add(okButton);
    buttonPane.add(cancelButton);
    okButton.addActionListener(this);
    cancelButton.addActionListener(this);

    getContentPane().add(buttonPane, "wrap");

    pack();
  }

  //
  // Handles events for the text field.
  //
  @Override
  public void actionPerformed(ActionEvent e) {
    if (OK_BUTTON_TEXT.equalsIgnoreCase(e.getActionCommand())) {

      StringBuffer logStr = new StringBuffer("new params selected: ");

      // collect settings
      this.session.countStrategy = this.optionPane.getSelectedStrategyValue();
      logStr.append("count strategy: ").append(this.optionPane.getSelectedStrategyValue());

      this.session.giAlgorithm = this.optionPane.getSelectedAlgorithmValue();
      logStr.append(", GI algorithm: ").append(this.optionPane.getSelectedAlgorithmValue());

      this.session.normalizationThreshold = this.optionPane.getNormalizationThreshold();
      logStr.append(", norm threshold: ").append(this.optionPane.getNormalizationThreshold());

      // the output file names
      this.session.grammarOutputFileName = this.optionPane.getGrammarOutputFileName();
      logStr.append(", GI output: ").append(this.optionPane.getGrammarOutputFileName());

      this.session.ruleDensityOutputFileName = this.optionPane.getRuleCoverageFileName();
      logStr.append(", rule density output: ").append(this.optionPane.getRuleCoverageFileName());

      this.session.anomaliesOutputFileName = this.optionPane.getAnomalyOutputFileName();
      logStr.append(", anomalies output: ").append(this.optionPane.getAnomalyOutputFileName());

      this.session.chartsSaveFolder = this.optionPane.getChartsFolderName();
      logStr.append(", charts output: ").append(this.optionPane.getChartsFolderName());

      LOGGER.info(logStr.toString());

    }
    else if (CANCEL_BUTTON_TEXT.equalsIgnoreCase(e.getActionCommand())) {
      assert true;
    }

    this.dispose();
  }

  /**
   * Clears the dialog and hides it.
   */
  public void clearAndHide() {
    setVisible(false);
  }
}