package net.seninp.grammarviz.view;

import java.awt.Dimension;
import java.awt.Point;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JFrame;
import javax.swing.JPanel;
import net.miginfocom.swing.MigLayout;
import net.seninp.grammarviz.session.UserSession;

/* 1.4 example used by DialogDemo.java. */
class GrammarvizGuesserDialog extends JDialog implements ActionListener {

  private static final long serialVersionUID = 8146102612457794550L;

  private static final String OK_BUTTON_TEXT = "OK";
  private static final String CANCEL_BUTTON_TEXT = "Cancel";

  //private UserSession session;

  private GrammarvizGuesserPane guesserPane;

  protected volatile boolean wasCancelled;

  /** Creates the reusable dialog. */
  public GrammarvizGuesserDialog(JFrame topFrame, JPanel guesserPane, UserSession session) {

    super(topFrame, true);

    if (topFrame != null) {
      Dimension parentSize = topFrame.getSize();
      Point p = topFrame.getLocation();
      setLocation(p.x + parentSize.width / 4, p.y + parentSize.height / 4);
    }

    this.setTitle("Sampler interval and parameter ranges verification");

    // this.session = session;

    this.guesserPane = (GrammarvizGuesserPane) guesserPane;

    MigLayout mainFrameLayout = new MigLayout("fill", "[grow,center]", "[grow]5[]");

    getContentPane().setLayout(mainFrameLayout);

    getContentPane().add(this.guesserPane, "h 200:200:,w 400:400:,growx,growy,wrap");

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

      // set params
      this.wasCancelled = false;

    }
    else if (CANCEL_BUTTON_TEXT.equalsIgnoreCase(e.getActionCommand())) {
      this.wasCancelled = true;
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