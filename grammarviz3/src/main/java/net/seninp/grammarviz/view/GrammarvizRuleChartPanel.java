package net.seninp.grammarviz.view;

import java.awt.BasicStroke;
import java.awt.Color;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.util.ArrayList;
import java.util.Arrays;
import javax.swing.JPanel;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import net.seninp.gi.logic.RuleInterval;
import net.seninp.grammarviz.controller.GrammarVizController;
import net.seninp.grammarviz.session.UserSession;
import net.seninp.jmotif.sax.TSProcessor;
import net.seninp.jmotif.sax.discord.DiscordRecord;
import net.seninp.util.StackTrace;

public class GrammarvizRuleChartPanel extends JPanel implements PropertyChangeListener {

  /** Fancy serial. */
  private static final long serialVersionUID = 5334407476500195779L;

  /** The chart container. */
  private JFreeChart chart;

  /** The plot itself. */
  private XYPlot plot;

  /** Current chart data instance. */
  private UserSession session;

  private TSProcessor tp;
  private GrammarVizController controller;

  /**
   * Constructor.
   */
  public GrammarvizRuleChartPanel() {
    super();
    tp = new TSProcessor();
  }

  /**
   * Adds a controler instance to get normalization value from.
   * 
   * @param controller the controller instance.
   */
  public void setController(GrammarVizController controller) {
    this.controller = controller;
  }

  /**
   * Create the chart for the original time series.
   * 
   * @return a JFreeChart object of the chart
   * @throws TSException
   */
  private void chartIntervals(ArrayList<double[]> intervals) throws Exception {

    // making the data
    //
    XYSeriesCollection collection = new XYSeriesCollection();
    int counter = 0;
    for (double[] series : intervals) {
      collection.addSeries(toSeries(counter++, series));
    }

    // set the renderer
    //
    XYLineAndShapeRenderer xyRenderer = new XYLineAndShapeRenderer(true, false);
    xyRenderer.setSeriesPaint(0, new Color(0, 0, 0));
    xyRenderer.setBaseStroke(new BasicStroke(3));

    // X - the time axis
    //
    NumberAxis timeAxis = new NumberAxis();

    // Y axis
    //
    NumberAxis valueAxis = new NumberAxis();

    // put these into collection of dots
    //
    this.plot = new XYPlot(collection, timeAxis, valueAxis, xyRenderer);

    // enable panning
    //
    this.plot.setDomainPannable(true);
    this.plot.setRangePannable(true);

    // finally, create the chart
    this.chart = new JFreeChart("", JFreeChart.DEFAULT_TITLE_FONT, plot, false);

    // and put it on the show
    ChartPanel chartPanel = new ChartPanel(this.chart);
    chartPanel.setMinimumDrawWidth(0);
    chartPanel.setMinimumDrawHeight(0);
    chartPanel.setMaximumDrawWidth(1920);
    chartPanel.setMaximumDrawHeight(1200);

    chartPanel.setMouseWheelEnabled(true);

    // cleanup all the content
    //
    this.removeAll();

    // put the chart on show
    //
    this.add(chartPanel);

    // not sure if I need this
    //
    this.validate();
    this.repaint();

  }

  /**
   * Converts an array to a normalized XYSeries to be digested with JFreeChart.
   * 
   * @param index
   * @param series
   * @return
   * @throws TSException
   */
  private XYSeries toSeries(int index, double[] series) throws Exception {
    double[] normalizedSubseries = tp.znorm(series, controller.getSession().normalizationThreshold);
    XYSeries res = new XYSeries("series" + String.valueOf(index));
    for (int i = 0; i < normalizedSubseries.length; i++) {
      res.add(i, normalizedSubseries[i]);
    }
    return res;
  }

  /**
   * Highlight the original time series sequences of a rule.
   * 
   * @param index index of the rule in the sequitur table.
   */
  protected void chartIntervalsForRule(ArrayList<String> newlySelectedRaw) {
    try {
      ArrayList<double[]> intervals = new ArrayList<double[]>();
      for (String str : newlySelectedRaw) {
        ArrayList<RuleInterval> arrPos = this.session.chartData
            .getRulePositionsByRuleNum(Integer.valueOf(str));
        for (RuleInterval saxPos : arrPos) {
          intervals.add(extractInterval(saxPos.getStart(), saxPos.getEnd()));
        }
      }
      chartIntervals(intervals);
    }
    catch (Exception e) {
      System.err.println(StackTrace.toString(e));
    }
  }

  /**
   * Highlight the original time series sequences of a sub-sequences class.
   * 
   * @param index index of the class in the sub-sequences class table.
   */
  protected void chartIntervalsForClass(ArrayList<String> newlySelectedRaw) {
    try {
      ArrayList<double[]> intervals = new ArrayList<double[]>();
      for (String str : newlySelectedRaw) {
          ArrayList<RuleInterval> arrPos = this.session.chartData
              .getSubsequencesPositionsByClassNum(Integer.valueOf(str));
          for (RuleInterval saxPos : arrPos) {
            intervals.add(extractInterval(saxPos.getStart(), saxPos.getEnd()));
          }
        }
        chartIntervals(intervals);
      }
      catch (Exception e) {
        System.err.println(StackTrace.toString(e));
      }
  }

  /**
   * Charts a subsequence for a selected row in the anomaly table.
   * 
   * @param newlySelectedAnomalies
   */
  private void chartIntervalForAnomaly(ArrayList<String> newlySelectedAnomalies) {
    try {
      ArrayList<double[]> intervals = new ArrayList<double[]>();
      for (String str : newlySelectedAnomalies) {
        DiscordRecord dr = this.session.chartData.getAnomalies().get(Integer.valueOf(str));
        intervals.add(extractInterval(dr.getPosition(), dr.getPosition() + dr.getLength()));
      }
      chartIntervals(intervals);
    }
    catch (Exception e) {
      System.err.println(StackTrace.toString(e));
    }
  }

  /**
   * Extracts a subsequence of the original time series.
   * 
   * @param startPos the start position.
   * @param endPos the end position.
   * @return the subsequence.
   * @throws Exception if error occurs.
   */
  private double[] extractInterval(int startPos, int endPos) throws Exception {
    if (this.session.chartData.getOriginalTimeseries().length <= (endPos - startPos)) {
      return Arrays.copyOf(this.session.chartData.getOriginalTimeseries(),
          this.session.chartData.getOriginalTimeseries().length);
    }
    return Arrays.copyOfRange(this.session.chartData.getOriginalTimeseries(), startPos, endPos);
  }

  /**
   * Clears the chart panel of the content.
   */
  public void clear() {
    this.removeAll();
    this.validate();
    this.repaint();
  }

  @Override
  public void propertyChange(PropertyChangeEvent evt) {

    if (GrammarRulesPanel.FIRING_PROPERTY.equalsIgnoreCase(evt.getPropertyName())) {
      @SuppressWarnings("unchecked")
      ArrayList<String> newlySelectedRaw = (ArrayList<String>) evt.getNewValue();
      chartIntervalsForRule(newlySelectedRaw);
    }
    else if (PackedRulesPanel.FIRING_PROPERTY_PACKED.equalsIgnoreCase(evt.getPropertyName())) {
      @SuppressWarnings("unchecked")
      ArrayList<String> newlySelectedRaw = (ArrayList<String>) evt.getNewValue();
      // chartIntervalsForRule(newlySelectedRaw);
      // String newlySelectedRaw = (String) evt.getNewValue();
      chartIntervalsForClass(newlySelectedRaw);

    }
    else if (RulesPeriodicityPanel.FIRING_PROPERTY_PERIOD.equalsIgnoreCase(evt.getPropertyName())) {
      String newlySelectedRaw = (String) evt.getNewValue();
      ArrayList<String> param = new ArrayList<String>(1);
      param.add(newlySelectedRaw);
      chartIntervalsForRule(param);
    }
    else if (GrammarVizAnomaliesPanel.FIRING_PROPERTY_ANOMALY
        .equalsIgnoreCase(evt.getPropertyName())) {
      @SuppressWarnings("unchecked")
      ArrayList<String> newlySelectedRaw = (ArrayList<String>) evt.getNewValue();
      chartIntervalForAnomaly(newlySelectedRaw);
    }

  }

  public void setChartData(UserSession session) {
    this.session = session;
  }

}
