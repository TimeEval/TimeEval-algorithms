package net.seninp.grammarviz.controller;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;
import java.util.Observable;
import javax.swing.JFileChooser;
import net.seninp.grammarviz.model.GrammarVizMessage;
import net.seninp.grammarviz.model.GrammarVizModel;
import net.seninp.grammarviz.session.UserSession;

/**
 * Implements the Controler component for GrammarViz2 GUI MVC.
 * 
 * @author psenin
 * 
 */
public class GrammarVizController extends Observable implements ActionListener {

  private GrammarVizModel model;

  private UserSession session;

  /**
   * Constructor.
   * 
   * @param model the program's model.
   */
  public GrammarVizController(GrammarVizModel model) {
    super();
    this.model = model;
    this.session = new UserSession();
  }

  /**
   * Implements a listener for the "Browse" button at GUI; opens FileChooser and so on.
   * 
   * @return the action listener.
   */
  public ActionListener getBrowseFilesListener() {

    ActionListener selectDataActionListener = new ActionListener() {

      public void actionPerformed(ActionEvent e) {

        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("Select Data File");

        String filename = model.getDataFileName();
        if (!((null == filename) || filename.isEmpty())) {
          fileChooser.setSelectedFile(new File(filename));
        }

        if (fileChooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
          File file = fileChooser.getSelectedFile();

          // here it calls to model -informing about the selected file.
          //
          model.setDataSource(file.getAbsolutePath());
        }
      }

    };
    return selectDataActionListener;
  }

  /**
   * Load file listener.
   * 
   * @return the listener instance.
   */
  public ActionListener getLoadFileListener() {
    ActionListener loadDataActionListener = new ActionListener() {
      public void actionPerformed(ActionEvent e) {
        model.loadData(e.getActionCommand());
      }
    };
    return loadDataActionListener;
  }

  /**
   * This provide Process action listener. Gets all the parameters from the session component
   * 
   * @return
   */
  public ActionListener getProcessDataListener() {

    ActionListener discretizeAndGrammarListener = new ActionListener() {
      public void actionPerformed(ActionEvent event) {

        StringBuffer logSB = new StringBuffer("running inference with settings:");

        logSB.append(" SAX window: ").append(session.useSlidingWindow);
        logSB.append(", SAX paa: ").append(session.useSlidingWindow);
        logSB.append(", SAX alphabet: ").append(session.useSlidingWindow);

        logSB.append(", sliding window:").append(session.useSlidingWindow);
        logSB.append(", num.reduction:").append(session.useSlidingWindow);
        logSB.append(", norm.threshold: ").append(session.useSlidingWindow);

        logSB.append(", GI alg: ").append(session.giAlgorithm);

        logSB.append(", grammar filename: ").append(session.useSlidingWindow);

        log(logSB.toString());

        try {
          model.processData(session.giAlgorithm, session.useSlidingWindow,
              session.numerosityReductionStrategy, session.saxWindow, session.saxPAA,
              session.saxAlphabet, session.normalizationThreshold, session.grammarOutputFileName);
        }
        catch (IOException exception) {
          // TODO Auto-generated catch block
          exception.printStackTrace();
        }

      }
    };
    return discretizeAndGrammarListener;
  }

  @Override
  public void actionPerformed(ActionEvent e) {
    this.setChanged();
    notifyObservers(new GrammarVizMessage(GrammarVizMessage.STATUS_MESSAGE,
        "controller: Unknown action performed " + e.getActionCommand()));
  }

  /**
   * Gets the current session.
   * 
   * @return
   */
  public UserSession getSession() {
    return this.session;
  }

  /**
   * Performs logging messages distribution.
   * 
   * @param message the message to log.
   */
  private void log(String message) {
    this.setChanged();
    notifyObservers(
        new GrammarVizMessage(GrammarVizMessage.STATUS_MESSAGE, "controller: " + message));
  }
}
