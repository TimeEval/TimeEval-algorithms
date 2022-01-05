package net.seninp.tinker;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.seninp.util.StackTrace;

public class MovieUtils {

  final static Charset DEFAULT_CHARSET = StandardCharsets.UTF_8;

  // static block - we instantiate the logger
  //
  private static final Logger LOGGER = LoggerFactory.getLogger(MovieMaker.class);

  /**
   * This reads the data
   * 
   * @param fname The filename.
   * @return
   */
  public static double[] loadData(String fname) {

    LOGGER.info("reading from " + fname);

    long lineCounter = 0;
    double ts[] = new double[1];

    Path path = Paths.get(fname);

    ArrayList<Double> data = new ArrayList<Double>();

    try {

      BufferedReader reader = Files.newBufferedReader(path, DEFAULT_CHARSET);

      String line = null;
      while ((line = reader.readLine()) != null) {
        String[] lineSplit = line.trim().split("\\s+");
        for (int i = 0; i < lineSplit.length; i++) {
          double value = new BigDecimal(lineSplit[i]).doubleValue();
          data.add(value);
        }
        lineCounter++;
      }
      reader.close();
    }
    catch (Exception e) {
      System.err.println(StackTrace.toString(e));
    }
    finally {
      assert true;
    }

    if (!(data.isEmpty())) {
      ts = new double[data.size()];
      for (int i = 0; i < data.size(); i++) {
        ts[i] = data.get(i);
      }
    }

    LOGGER.info("loaded " + data.size() + " points from " + lineCounter + " lines in " + fname);
    return ts;

  }

  public static void saveColumn(int[] density, String filename) throws IOException {
    BufferedWriter bw = new BufferedWriter(new FileWriter(new File(filename)));
    for (int n : density) {
      bw.write(Integer.valueOf(n).toString() + "\n");
    }
    bw.close();
  }
}
