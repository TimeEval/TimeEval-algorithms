package net.seninp.grammarviz.multivariate.configuration;

import java.util.ArrayList;
import java.util.List;

import com.beust.jcommander.Parameter;
import net.seninp.grammarviz.gi.GIAlgorithm;
import net.seninp.grammarviz.anomaly.AnomalyAlgorithm;
import net.seninp.grammarviz.multivariate.configuration.MultivariateStrategy;
import net.seninp.grammarviz.multivariate.configuration.OutputMode;
import net.seninp.jmotif.sax.NumerosityReductionStrategy;

/**
 * Implements a parameters template for CLI conversion.
 *
 * @author psenin
 */
public class GrammarVizAnomalyParameters {

    // general setup
    //
    @Parameter
    public List<String> parameters = new ArrayList<>();

    @Parameter(names = {"--help", "-h"}, help = true)
    public boolean help;

    // dataset
    //
    @Parameter(names = {"--data", "-i"}, description = "The input file name")
    public String IN_FILE;

    // output
    //
    @Parameter(names = {"--output", "-o"}, description = "The output file prefix")
    public String OUT_FILE = "";
    public String DISTANCE_FILENAME = OUT_FILE + "_distances.txt";

    // discretization parameters
    //
    @Parameter(names = {"--window_size", "-w"}, description = "Sliding window size")
    public int SAX_WINDOW_SIZE = 170;

    @Parameter(names = {"--word_size", "-p"}, description = "PAA word size")
    public int SAX_PAA_SIZE = 4;

    @Parameter(names = {"--alphabet_size", "-a"}, description = "SAX alphabet size")
    public int SAX_ALPHABET_SIZE = 4;

    @Parameter(names = "--strategy", description = "Numerosity reduction strategy")
    public NumerosityReductionStrategy SAX_NR_STRATEGY = NumerosityReductionStrategy.EXACT;

    @Parameter(names = "--threshold", description = "Normalization threshold")
    public double SAX_NORM_THRESHOLD = 0.01;

    // the algorithms params
    //
    @Parameter(names = {"--algorithm", "-alg"}, description = "The algorithm to use")
    public AnomalyAlgorithm ALGORITHM = AnomalyAlgorithm.RRA;

    @Parameter(names = {"--multivariate_strategy", "-multi"}, description = "The net.seninp.grammarviz.multivariate strategy to use")
    public MultivariateStrategy MULTIVARIATE_STRATEGY = MultivariateStrategy.SEPARATE_MAX;

    @Parameter(names = {"--output_mode", "-mode"}, description = "The algorithm to use to create the output")
    public OutputMode OUTPUT_MODE = OutputMode.FULL;

    @Parameter(names = {"--discords_num", "-n"}, description = "The algorithm to use")
    public int DISCORDS_NUM = 5;

    // GI parameter
    //
    @Parameter(names = {"--gi", "-g"}, description = "GI algorithm to use")
    public GIAlgorithm GI_ALGORITHM_IMPLEMENTATION = GIAlgorithm.SEQUITUR;

    // sub-sampling parameter
    //
    @Parameter(names = {
            "--subsample"}, description = "RRASAMPLED subsampling fraction (0.0 - 1.0) for longer time series")
    public Double SUBSAMPLING_FRACTION = Double.NaN;

    // grid boundaries for discretization parameters
    //
    @Parameter(names = {"--bounds",
            "-b"}, description = "RRASAMPLED grid boundaries (Wmin Wmax Wstep Pmin Pmax Pstep Amin Amax Astep)")
    public String GRID_BOUNDARIES = "10 100 10 10 50 10 2 12 2";

    // Random number generator seed parameter
    //
    @Parameter(names = {"--seed"}, description = "Random number generator seed")
    public long RANDOM_SEED = 42L;

}
