package net.seninp.grammarviz.view;

import java.awt.Color;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.text.DecimalFormat;
import java.util.concurrent.atomic.AtomicBoolean;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.IntervalMarker;
import org.jfree.chart.plot.Marker;
import org.jfree.chart.plot.XYPlot;
import org.jfree.ui.Layer;

public class MouseMarker extends MouseAdapter {

  private static AtomicBoolean isMarking = new AtomicBoolean(false);

  private static final DecimalFormat dfFormatter = (new DecimalFormat("0.00"));

  private Marker marker;
  private Double markerStart = Double.NaN;
  private Double markerEnd = Double.NaN;

  private final XYPlot plot;

  private final JFreeChart chart;

  private final ChartPanel panel;

  private Object selectionLock;

  public MouseMarker(ChartPanel panel) {
    this.panel = panel;
    this.chart = panel.getChart();
    this.plot = (XYPlot) chart.getPlot();
  }

  private void updateMarker() {
    if (marker != null) {
      plot.removeDomainMarker(marker, Layer.BACKGROUND);
    }
    if (!(markerStart.isNaN() && markerEnd.isNaN())) {
      if (markerEnd > markerStart) {
        marker = new IntervalMarker(markerStart, markerEnd);
        marker.setPaint(new Color(0xDD, 0xFF, 0xDD, 0x90));
        marker.setAlpha(0.7f);
        plot.addDomainMarker(marker, Layer.BACKGROUND);
      }
      else {
        marker = new IntervalMarker(markerEnd, markerStart);
        marker.setPaint(new Color(0xDD, 0xFF, 0xDD, 0x90));
        marker.setAlpha(0.7f);
        plot.addDomainMarker(marker, Layer.BACKGROUND);
      }
    }
  }

  private Double getPosition(MouseEvent e) {
    // System.err.println("Event position " + e.getX() + ", " + e.getY());
    Point2D p = panel.translateScreenToJava2D(e.getPoint());
    Rectangle2D plotArea = panel.getScreenDataArea();
    XYPlot plot = (XYPlot) chart.getPlot();
    return plot.getDomainAxis().java2DToValue(p.getX(), plotArea, plot.getDomainAxisEdge());
  }

  @Override
  public void mouseDragged(MouseEvent e) {
    if (isMarking.get()) {
      markerEnd = getPosition(e);
      updateMarker();
    }
  }

  @Override
  public void mouseReleased(MouseEvent e) {
    isMarking.set(false);
    markerEnd = getPosition(e);
    updateMarker();
    synchronized (selectionLock) {
      selectionLock.notifyAll();
    }
  }

  @Override
  public void mousePressed(MouseEvent e) {
    isMarking.set(true);
    markerStart = getPosition(e);
  }

  public void setLockObject(Object selectionLock) {
    this.selectionLock = selectionLock;
  }

  public String getIntervalStr() {
    return dfFormatter.format(this.markerStart) + " - " + dfFormatter.format(this.markerEnd);
  }

  public double getSelectionStart() {
    if (markerEnd >= markerStart) {
      return this.markerStart;
    }
    return this.markerEnd;
  }

  public double getSelectionEnd() {
    if (markerEnd >= markerStart) {
      return this.markerEnd;
    }
    return this.markerStart;
  }
}