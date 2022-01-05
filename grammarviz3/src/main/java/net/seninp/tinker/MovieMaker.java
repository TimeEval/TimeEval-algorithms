package net.seninp.tinker;

import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.seninp.gi.logic.GrammarRuleRecord;
import net.seninp.gi.logic.GrammarRules;
import net.seninp.gi.logic.RuleInterval;
import net.seninp.gi.sequitur.SAXRule;
import net.seninp.gi.sequitur.SAXTerminal;
import net.seninp.gi.sequitur.SequiturFactory;
import net.seninp.jmotif.sax.TSProcessor;
import net.seninp.jmotif.sax.alphabet.NormalAlphabet;
import net.seninp.jmotif.sax.datastructure.SAXRecords;

public class MovieMaker {

  // locale, charset, logger, etc
  //
  final static Charset DEFAULT_CHARSET = StandardCharsets.UTF_8;

  // data file
  //
  private static final String DATA_FILENAME = "data/asys40.txt";
  private static final String OUT_PREFIX = "movie/density";

  // params
  //
  private static final int WINDOW_SIZE = 400;
  private static final int PAA_SIZE = 8;
  private static final int A_SIZE = 6;
  private static final double NORMALIZATION_THRESHOLD = 0.01D;

  private static final NormalAlphabet normalA = new NormalAlphabet();

  // data
  //
  private static double[] ts;
  private static TSProcessor tp = new TSProcessor();

  // static block - we instantiate the logger
  //
  private static final Logger LOGGER = LoggerFactory.getLogger(MovieMaker.class);

  // the main runnable
  //
  public static void main(String[] args) throws Exception {

    // load the data
    //
    ts = MovieUtils.loadData(DATA_FILENAME);

    // we keep discretized data here
    //
    SAXRecords saxFrequencyData = new SAXRecords();

    // getting ready
    //
    SAXRule.reset();
    SAXRule grammar = new SAXRule();
    String previousString = "";

    // scan across the time series extract sub sequences, and convert
    // them to strings
    int stringPosCounter = 0;
    int saveFileCounter = 0;
    for (int i = 0; i < ts.length - (WINDOW_SIZE - 1); i++) {

      if (i % (1000) == 0) {
        LOGGER.info("processing position " + i + " out of " + ts.length);
      }

      // fix the current subsection
      double[] subSection = Arrays.copyOfRange(ts, i, i + WINDOW_SIZE);

      // Z normalize it
      subSection = tp.znorm(subSection, NORMALIZATION_THRESHOLD);

      // perform PAA conversion if needed
      double[] paa = tp.paa(subSection, PAA_SIZE);

      // Convert the PAA to a string.
      char[] currentString = tp.ts2String(paa, normalA.getCuts(A_SIZE));

      // NumerosityReduction
      if (!previousString.isEmpty()
          && previousString.equalsIgnoreCase(String.valueOf(currentString))) {
        continue;
      }
      previousString = String.valueOf(currentString);

      // add a terminal to the Sequitur
      //
      grammar.last().insertAfter(new SAXTerminal(String.valueOf(currentString), stringPosCounter));
      grammar.last().p.check();

      // add the word to frequency data structure
      //
      saxFrequencyData.add(currentString, i);

      // save the current rule density curve
      //
      if (i >= WINDOW_SIZE && i < ts.length - WINDOW_SIZE * 2) {

        // index sax words
        //
        saxFrequencyData.buildIndex();

        // convert the grammar to a simple data structure
        //
        GrammarRules rules = grammar.toGrammarRulesData();

        // and populate the coverage
        //
        SequiturFactory.updateRuleIntervals(rules, saxFrequencyData, true,
            Arrays.copyOfRange(ts, i, i + WINDOW_SIZE), WINDOW_SIZE, PAA_SIZE);

        // collect the coverage
        //
        int[] coverageArray = new int[i + WINDOW_SIZE];
        for (GrammarRuleRecord r : rules) {
          if (0 == r.ruleNumber()) {
            continue;
          }
          ArrayList<RuleInterval> arrPos = r.getRuleIntervals();
          for (RuleInterval saxPos : arrPos) {
            int startPos = saxPos.getStart();
            int endPos = saxPos.getEnd();
            for (int j = startPos; j < endPos; j++) {
              coverageArray[j] = coverageArray[j] + 1;
            }
          }
        }

        String outFname = OUT_PREFIX + String.format("%04d", saveFileCounter) + ".csv";
        MovieUtils.saveColumn(coverageArray, outFname);

      }

      // moving on...
      //
      stringPosCounter++;
      saveFileCounter++;
    }

    // String cmdLine = "Rscript RCode/movie_plotter.R " + DATA_FILENAME + " " + outFname
    // + " movie/" + String.format("%04d", counter) + ".jpg";
    // consoleLogger.info(cmdLine);
    // Runtime r = Runtime.getRuntime();
    // Process p = r.exec(cmdLine);
    // p.waitFor();
    // BufferedReader b = new BufferedReader(new InputStreamReader(p.getInputStream()));
    // String line = "";
    //
    // while ((line = b.readLine()) != null) {
    // System.out.println(line);
    // }
    //
    // b.close();

  }
}
