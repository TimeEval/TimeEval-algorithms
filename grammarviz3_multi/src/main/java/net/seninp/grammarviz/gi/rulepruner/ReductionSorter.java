package net.seninp.grammarviz.gi.rulepruner;

import net.seninp.grammarviz.gi.rulepruner.SampledPoint;

import java.util.Comparator;

public class ReductionSorter implements Comparator<SampledPoint> {

  @Override
  public int compare(SampledPoint o1, SampledPoint o2) {
    return Double.compare(o1.getReduction(), o2.getReduction());
  }

}
