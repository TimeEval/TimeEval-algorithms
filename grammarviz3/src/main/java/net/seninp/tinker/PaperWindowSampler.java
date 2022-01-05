package net.seninp.tinker;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.seninp.gi.GIAlgorithm;
import net.seninp.gi.rulepruner.ReductionSorter;
import net.seninp.gi.rulepruner.RulePruner;
import net.seninp.gi.rulepruner.SampledPoint;
import net.seninp.jmotif.sax.NumerosityReductionStrategy;
import net.seninp.jmotif.sax.TSProcessor;

public class PaperWindowSampler {

  private static final String IN_DATA = "RCode/TKDD/sine_and_5anomalies.txt";

  private static final int SAMPLE_START = 745;
  private static final int SAMPLE_END = 1070;
  private static final int SAMPLE_STEP = 5;

  private static final TSProcessor tp = new TSProcessor();

  private static final Logger LOGGER = LoggerFactory.getLogger(PaperWindowSampler.class);

  public static void main(String[] args) throws Exception {

    BufferedWriter bw = new BufferedWriter(new FileWriter(new File("RCode/TKDD/sine_and_5anomalies_out.txt")));

    // HashMap<String, SampledPoint> res_global = new HashMap<String, SampledPoint>();

    double[] ts = TSProcessor.readFileColumn(IN_DATA, 0, 0);
    LOGGER.info("read " + ts.length + " points from " + IN_DATA);

    for (int upperLimit = 10; upperLimit <= (SAMPLE_END
        - SAMPLE_START); upperLimit += SAMPLE_STEP) {

      ArrayList<SampledPoint> res = new ArrayList<SampledPoint>();

      LOGGER.info("sampling interval " + SAMPLE_START + " - " + (SAMPLE_START + upperLimit));

      double[] sampledTS = tp.subseriesByCopy(ts, SAMPLE_START, (SAMPLE_START + upperLimit));
      RulePruner rp = new RulePruner(sampledTS);

      for (int ws = 10; ws <= upperLimit; ws = ws + 1) {

        for (int ps = 2; ps < 12; ps = ps + 1) {

          // check for invalid cases
          if (ps > ws) {
            continue;
          }

          for (int as = 2; as < 12; as = as + 1) {

            SampledPoint p = null;

            p = rp.sample(ws, ps, as, GIAlgorithm.REPAIR, NumerosityReductionStrategy.NONE, 0.001);

            if (null != p) {
              res.add(p);
            }

          }
        }
      }

      Collections.sort(res, new ReductionSorter());
      SampledPoint best_point = res.get(0);

      String str = SAMPLE_START + "," + (SAMPLE_START + upperLimit) + ","
          + best_point.toLogString();
      
      System.out.println("*** " + str);
      bw.write(str + "\n");
    }

    bw.close();
  }

}
