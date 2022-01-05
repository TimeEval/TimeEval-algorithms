package net.seninp.tinker;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.supercsv.io.CsvListReader;
import org.supercsv.prefs.CsvPreference;

public class ParamSampler {

  private static final String prefix = "/media/Stock/tmp/ydata-labeled-time-series-anomalies-v1_0/A4Benchmark/";
  private static final String fileExtension = ".csv";

  // static block - we instantiate the logger
  //
  private static final Logger LOGGER = LoggerFactory.getLogger(ParamSampler.class);

  // the main runnable
  //
  public static void main(String[] args) throws Exception {

    File dir = new File(prefix);
    File[] filesList = dir.listFiles(new FilenameFilter() {
      public boolean accept(File dir, String name) {
        return name.toLowerCase().endsWith(fileExtension);
      }
    });

    List<String> samplerBatch = new ArrayList<String>();

    // this runs for each file
    //
    for (File file : filesList) {
      if (file.isFile()) {

        // get the file reader set up
        //
        LOGGER.info("processing " + file.getName());
        CsvListReader reader = new CsvListReader(new FileReader(file),
            CsvPreference.STANDARD_PREFERENCE);
        final String[] header = reader.getHeader(true);
        LOGGER.info(" file header: " + Arrays.toString(header));

        // findout needed fields indexes
        //
        int valueIdx = -1;
        int anomalyFlagIdx = -1;
        for (int i = 0; i < header.length; i++) {
          String str = header[i];
          if (str.equalsIgnoreCase("value")) {
            valueIdx = i;
          }
          else if (str.equalsIgnoreCase("anomaly")) {
            anomalyFlagIdx = i;
          }
        }

        // setup data keepers
        //
        List<Double> values = new ArrayList<Double>();
        List<Byte> anomalyFlags = new ArrayList<Byte>();

        // setup the processor
        //
        List<String> record;
        while ((record = reader.read()) != null) {

          Double value = Double.valueOf(record.get(valueIdx)).doubleValue();
          values.add(value);

          Byte isAnomaly = Byte.valueOf(record.get(anomalyFlagIdx));
          anomalyFlags.add(isAnomaly);

        }
        reader.close();

        // write the data file
        //
        String samplerInputFilename = file.getName().concat(".column");
        BufferedWriter bw = new BufferedWriter(
            new FileWriter(new File(prefix + samplerInputFilename)));
        for (Double v : values) {
          bw.write(v + "\n");
        }
        bw.close();

        // set boundaries string
        //
        StringBuffer samplingBoundaries = new StringBuffer("10 ");
        if (values.size() < 3000) {
          samplingBoundaries.append(Integer.valueOf(values.size() / 10).toString());
        }
        else {
          samplingBoundaries.append("300 ");
        }
        samplingBoundaries.append(" 10 2 20 1 2 10 1");

        // makeup the sampler command
        //
        StringBuffer samplerCommand = new StringBuffer(
            "java -Xmx4G -cp \"jmotif-gi-0.3.1-SNAPSHOT-jar-with-dependencies.jar\"");
        samplerCommand.append(" net.seninp.gi.rulepruner.RulePrunerPrinter ");

        samplerCommand.append(" -d ");
        samplerCommand.append(samplerInputFilename);

        samplerCommand.append(" -b \"");
        samplerCommand.append(samplingBoundaries.toString()).append("\"");

        samplerCommand.append(" -o ");
        samplerCommand.append(samplerInputFilename).append(".out");

        samplerBatch.add(samplerCommand.toString());

      }
    }

    for (String line : samplerBatch) {
      System.out.println(line);
    }

  }

  // /**
  // * Sets up the processors.
  // *
  // * @return the cell processors
  // */
  // private static CellProcessor[] getProcessors() {
  //
  // final CellProcessor[] processors = new CellProcessor[] { new ParseDouble(), // timestamp
  // new ParseDouble(), // value
  // new ParseInt() // anomaly flag
  // };
  //
  // return processors;
  // }

}
