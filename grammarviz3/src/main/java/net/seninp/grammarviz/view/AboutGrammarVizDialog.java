package net.seninp.grammarviz.view;

import java.awt.Dimension;
import java.awt.Point;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;
import javax.swing.JButton;
import javax.swing.JDialog;
import javax.swing.JEditorPane;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import net.miginfocom.swing.MigLayout;

public class AboutGrammarVizDialog extends JDialog implements ActionListener {

  private static final long serialVersionUID = -8273240552350932580L;

  private static final String OK_BUTTON_TEXT = "OK";

  public AboutGrammarVizDialog(JFrame parentFrame) {

    super(parentFrame, true);
    if (parentFrame != null) {
      Dimension parentSize = parentFrame.getSize();
      Point p = parentFrame.getLocation();
      setLocation(p.x + parentSize.width / 4, p.y + parentSize.height / 4);
    }

    JEditorPane aboutTextPane = new JEditorPane();

    aboutTextPane.setEditable(false);
    java.net.URL helpURL = AboutGrammarVizDialog.class.getResource("/AboutText.html");
    if (helpURL != null) {
      try {
        aboutTextPane.setPage(helpURL);
      }
      catch (IOException e) {
        System.err.println("Attempted to read a bad URL: " + helpURL);
      }
    }
    else {
      System.err.println("Couldn't find file: AboutText.html");
    }

    // Put the editor pane in a scroll pane.
    JScrollPane editorScrollPane = new JScrollPane(aboutTextPane);
    editorScrollPane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);

    MigLayout mainFrameLayout = new MigLayout("fill", "[grow,center]", "[grow]5[]");

    getContentPane().setLayout(mainFrameLayout);

    getContentPane().add(editorScrollPane, "h 200:300:,w 400:500:,growx,growy,wrap");

    JPanel buttonPane = new JPanel();
    JButton okButton = new JButton(OK_BUTTON_TEXT);

    buttonPane.add(okButton);
    okButton.addActionListener(this);

    getContentPane().add(buttonPane, "wrap");

    pack();
    setVisible(true);

  }

  @Override
  public void actionPerformed(ActionEvent e) {
    if (OK_BUTTON_TEXT.equalsIgnoreCase(e.getActionCommand())) {
      this.dispose();
    }
  }

  /** This method clears the dialog and hides it. */
  public void clearAndHide() {
    setVisible(false);
  }
}
