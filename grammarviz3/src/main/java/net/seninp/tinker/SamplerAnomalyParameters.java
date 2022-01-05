package net.seninp.tinker;

import java.util.ArrayList;
import java.util.List;
import com.beust.jcommander.Parameter;
import net.seninp.gi.GIAlgorithm;
import net.seninp.jmotif.sax.NumerosityReductionStrategy;

/**
 * Parameters accepted by the bitmap printer and their default values.
 * 
 * @author psenin
 * 
 */
public class SamplerAnomalyParameters {

  // general setup
  //
  @Parameter
  public List<String> parameters = new ArrayList<String>();

  // dataset
  //
  @Parameter(names = { "--data", "-d" }, description = "The input file name")
  public static String IN_FILE;

  // output
  //
  @Parameter(names = { "--out", "-o" }, description = "The output file name")
  public static String OUT_FILE;

  @Parameter(names = { "--dnum", "-n" }, description = "The number of discords to report")
  public static int DISCORDS_NUM = 5;

  // discretization parameters
  //
  @Parameter(names = { "--window_size", "-w" }, description = "SAX sliding window size")
  public static int SAX_WINDOW_SIZE = 30;

  @Parameter(names = { "--word_size", "-p" }, description = "SAX PAA word size")
  public static int SAX_PAA_SIZE = 4;

  @Parameter(names = { "--alphabet_size", "-a" }, description = "SAX alphabet size")
  public static int SAX_ALPHABET_SIZE = 3;

  @Parameter(names = "--strategy", description = "SAX numerosity reduction strategy")
  public static NumerosityReductionStrategy SAX_NR_STRATEGY = NumerosityReductionStrategy.NONE;

  @Parameter(names = "--threshold", description = "SAX normalization threshold")
  public static double SAX_NORM_THRESHOLD = 0.5;

  // GI parameter
  //
  @Parameter(names = { "--algorithm" }, description = "algorithm to use")
  public static GIAlgorithm GI_ALGORITHM_IMPLEMENTATION = GIAlgorithm.SEQUITUR;

}
