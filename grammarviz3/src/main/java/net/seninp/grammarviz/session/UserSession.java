package net.seninp.grammarviz.session;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import net.seninp.gi.GIAlgorithm;
import net.seninp.grammarviz.logic.CoverageCountStrategy;
import net.seninp.grammarviz.logic.GrammarVizChartData;
import net.seninp.jmotif.sax.NumerosityReductionStrategy;
import net.seninp.util.StackTrace;

/**
 * Keeps user parameters organized.
 * 
 * @author psenin
 * 
 */
public class UserSession {

  /** Params change event. */
  public static final String PARAMS_CHANGED_EVENT = "parameters_changed";

  /** The default values. */
  private static final int DEFAULT_SAX_WINDOW = 170;
  private static final int DEFAULT_SAX_PAA = 4;
  private static final int DEFAULT_SAX_ALPHABET = 4;
  private static final boolean USE_SLIDING_WINDOW = true;
  private static final Double DEFAULT_NORMALIZATION_THRESHOLD_VALUE = 0.05;
  private static final NumerosityReductionStrategy DEFAULT_NUMEROSITY_REDUCTION_STRATEGY = NumerosityReductionStrategy.EXACT;
  private static final GIAlgorithm DEFAULT_GI_ALGORITHM = GIAlgorithm.SEQUITUR;
  private static final CoverageCountStrategy DEFAULT_COUNT_STRATEGY = CoverageCountStrategy.COUNT;

  // discretization variables
  //
  public volatile int saxWindow;
  public volatile int saxPAA;
  public volatile int saxAlphabet;
  public volatile boolean useSlidingWindow;
  public volatile NumerosityReductionStrategy numerosityReductionStrategy;
  public volatile Double normalizationThreshold;

  // core algorithms variables
  //
  public volatile CoverageCountStrategy countStrategy;
  public volatile GIAlgorithm giAlgorithm;

  // guesser parameters
  //
  public volatile Integer samplingStart;
  public volatile Integer samplingEnd;
  public volatile Double minimalCoverThreshold = 0.98;
  public volatile int[] boundaries = { 10, 200, 10, 2, 10, 1, 2, 10, 1 };

  // auxiliary variables
  //
  public volatile String ruleDensityOutputFileName;
  public volatile String grammarOutputFileName;
  public volatile String anomaliesOutputFileName;
  public volatile String chartsSaveFolder;

  // listeners for the parameter changes
  //
  private ArrayList<ActionListener> listeners = new ArrayList<ActionListener>();

  // the huge class holding the timeseries and the grammar
  //
  public volatile GrammarVizChartData chartData;

  public UserSession() {

    super();

    this.saxWindow = DEFAULT_SAX_WINDOW;
    this.saxPAA = DEFAULT_SAX_PAA;
    this.saxAlphabet = DEFAULT_SAX_ALPHABET;
    this.useSlidingWindow = USE_SLIDING_WINDOW;
    this.numerosityReductionStrategy = DEFAULT_NUMEROSITY_REDUCTION_STRATEGY;
    this.normalizationThreshold = DEFAULT_NORMALIZATION_THRESHOLD_VALUE;
    this.giAlgorithm = DEFAULT_GI_ALGORITHM;
    this.countStrategy = DEFAULT_COUNT_STRATEGY;

    // attempt to fill the rule coverage name automatically
    //
    String filename = "";
    try {
      String currentPath = new File(".").getCanonicalPath();
      filename = currentPath + File.separator + "density_curve.txt";
    }
    catch (IOException e) {
      System.err.println(
          "Error has been thrown, unable to findout the current path: " + StackTrace.toString(e));
    }
    this.ruleDensityOutputFileName = filename;
  }

  /**
   * Adds a parameters change action listener which will be notified on any change.
   * 
   * @param e
   */
  public void addActionListener(ActionListener e) {
    this.listeners.add(e);
  }

  /**
   * Notifies all registered listeners.
   */
  public void notifyParametersChangeListeners() {
    ActionEvent event = new ActionEvent(this, 0, PARAMS_CHANGED_EVENT);
    for (ActionListener listener : listeners) {
      listener.actionPerformed(event);
    }

  }

}
