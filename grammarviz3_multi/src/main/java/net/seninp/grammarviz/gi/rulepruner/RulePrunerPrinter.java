package net.seninp.grammarviz.gi.rulepruner;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import com.beust.jcommander.JCommander;
import net.seninp.grammarviz.gi.rulepruner.ReductionSorter;
import net.seninp.grammarviz.gi.rulepruner.RulePruner;
import net.seninp.grammarviz.gi.rulepruner.SampledPoint;
import net.seninp.jmotif.sax.TSProcessor;
import net.seninp.util.StackTrace;

/**
 * Rule pruner experimentation.
 * 
 * @author psenin
 * 
 */
public class RulePrunerPrinter {

  //
  //
  // -b "10 200 10 2 10 1 2 10 1" -d
  // /media/Stock/tmp/ydata-labeled-time-series-anomalies-v1_0/A1Benchmark/real_22.csv.column -o
  // /media/Stock/tmp/test.out
  //
  //

  // constants and formatter
  //
  // private static final String COMMA = ",";
  private static final String CR = "\n";
  // private static final DecimalFormat dfPercent = (new DecimalFormat("0.00"));
  // private static final DecimalFormat dfSize = (new DecimalFormat("#.0000"));

  private static final String OUTPUT_HEADER = "window,paa,alphabet,approxDist,grammarSize,grammarRules,"
      + "compressedGrammarSize,prunedRules,isCovered,coverage\n";

  // the logger
  //
  // private static final Logger LOGGER = LoggerFactory.getLogger(RulePrunerPrinter.class);

  /**
   * Main runnable.
   * 
   * @param args parameters used.
   * @throws Exception if error occurs.
   */
  public static void main(String[] args) throws Exception {

    try {

      RulePrunerParameters params = new RulePrunerParameters();
      JCommander jct = new JCommander(params);

      if (0 == args.length) {
        jct.usage();
      }
      else {
        
        jct.parse(args);

        // get params printed
        //
        StringBuffer sb = new StringBuffer(1024);
        sb.append("Rule pruner CLI v.1").append(CR);
        sb.append("parameters:").append(CR);

        sb.append("  input file:           ").append(RulePrunerParameters.IN_FILE).append(CR);
        sb.append("  output file:          ").append(RulePrunerParameters.OUT_FILE).append(CR);
        sb.append("  SAX num. reduction:   ").append(RulePrunerParameters.SAX_NR_STRATEGY)
            .append(CR);
        sb.append("  SAX norm. threshold:  ").append(RulePrunerParameters.SAX_NORM_THRESHOLD)
            .append(CR);
        sb.append("  GI Algorithm:         ")
            .append(RulePrunerParameters.GI_ALGORITHM_IMPLEMENTATION).append(CR);
        sb.append("  Grid boundaries:      ").append(RulePrunerParameters.GRID_BOUNDARIES)
            .append(CR);

        if (!(Double.isNaN(RulePrunerParameters.SUBSAMPLING_FRACTION))) {
          sb.append("  Subsampling fraction: ").append(RulePrunerParameters.SUBSAMPLING_FRACTION)
              .append(CR);
        }

        // printer out the params before starting
        System.err.print(sb.toString());

        // read the data in
        String dataFName = RulePrunerParameters.IN_FILE;
        double[] ts = TSProcessor.readFileColumn(dataFName, 0, 0);
        if (!(Double.isNaN(RulePrunerParameters.SUBSAMPLING_FRACTION))) {
          ts = Arrays.copyOfRange(ts, 0,
              (int) Math.round((double) ts.length * RulePrunerParameters.SUBSAMPLING_FRACTION));
        }

        // printer out the params before starting
        System.err.println("  working with series of " + ts.length + " points ... " + CR);

        // parse the boundaries params
        int[] boundaries = toBoundaries(RulePrunerParameters.GRID_BOUNDARIES);

        // create the output file
        BufferedWriter bw = new BufferedWriter(
            new FileWriter(new File(RulePrunerParameters.OUT_FILE)));
        bw.write(OUTPUT_HEADER);

        ArrayList<SampledPoint> res = new ArrayList<SampledPoint>();

        // we need to use this in the loop
        RulePruner rp = new RulePruner(ts);

        // iterate over the grid evaluating the grammar
        //
        for (int WINDOW_SIZE = boundaries[0]; WINDOW_SIZE < boundaries[1]; WINDOW_SIZE += boundaries[2]) {
          for (int PAA_SIZE = boundaries[3]; PAA_SIZE < boundaries[4]; PAA_SIZE += boundaries[5]) {

            // check for invalid cases
            if (PAA_SIZE > WINDOW_SIZE) {
              continue;
            }

            for (int ALPHABET_SIZE = boundaries[6]; ALPHABET_SIZE < boundaries[7]; ALPHABET_SIZE += boundaries[8]) {

              SampledPoint p = rp.sample(WINDOW_SIZE, PAA_SIZE, ALPHABET_SIZE,
                  RulePrunerParameters.GI_ALGORITHM_IMPLEMENTATION,
                  RulePrunerParameters.SAX_NR_STRATEGY, RulePrunerParameters.SAX_NORM_THRESHOLD);

              bw.write(p.toLogString() + "\n");

              res.add(p);
            }
          }
        }

        bw.close();

        Collections.sort(res, new ReductionSorter());

        System.out.println("\nApparently, the best parameters are " + res.get(0).toString());

      }
    }
    catch (Exception e) {
      System.err.println("error occured while parsing parameters " + Arrays.toString(args) + CR
          + StackTrace.toString(e));
      System.exit(-1);
    }

  }

  /**
   * Converts a param string to boundaries array.
   * 
   * @param str
   * @return
   */
  private static int[] toBoundaries(String str) {
    int[] res = new int[9];
    String[] split = str.split("\\s+");
    for (int i = 0; i < 9; i++) {
      res[i] = Integer.valueOf(split[i]).intValue();
    }
    return res;
  }

}
