package net.seninp.grammarviz.cli;

import com.beust.jcommander.Parameter;
import net.seninp.jmotif.sax.NumerosityReductionStrategy;

import java.util.ArrayList;
import java.util.List;

/**
 * Implements a parameters template for CLI conversion.
 * 
 * @author psenin
 */
public class TS2GrammarParameters {

  // general setup
  @Parameter
  public List<String> parameters = new ArrayList<>();

  @Parameter(names = { "--help", "-h" }, help = true)
  public boolean help;

  // dataset
  @Parameter(names = { "--data_in", "-d" }, description = "The input file name")
  public static String IN_FILE;

  // output
  @Parameter(names = { "--data_out", "-o" }, description = "The output file name")
  public static String OUT_FILE;

  // discretization parameters
  @Parameter(names = { "--window_size", "-w" }, description = "Sliding window size")
  public static int SAX_WINDOW_SIZE = 30;

  @Parameter(names = { "--word_size", "-p" }, description = "PAA word size")
  public static int SAX_PAA_SIZE = 6;

  @Parameter(names = { "--alphabet_size", "-a" }, description = "SAX alphabet size")
  public static int SAX_ALPHABET_SIZE = 4;

  @Parameter(names = "--strategy", description = "Numerosity reduction strategy")
  public static NumerosityReductionStrategy SAX_NR_STRATEGY = NumerosityReductionStrategy.NONE;

  @Parameter(names = "--threshold", description = "Normalization threshold")
  public static double SAX_NORM_THRESHOLD = 0.01;

  @Parameter(names = "--prune", description = "Pass to prune rules")
  public static boolean PRUNE_RULES = false;

  @Parameter(names = {"--num-workers", "-n"}, description = "Number of worker threads to use for SAX")
  public static int NUM_WORKERS = 1;
}
