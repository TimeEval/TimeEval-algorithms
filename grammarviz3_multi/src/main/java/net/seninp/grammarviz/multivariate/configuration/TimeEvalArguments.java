package net.seninp.grammarviz.multivariate.configuration;

import com.google.gson.Gson;
import com.google.gson.annotations.SerializedName;
import net.seninp.jmotif.sax.NumerosityReductionStrategy;

public final class TimeEvalArguments {

    public String dataInput;
    public String dataOutput;
    public String executionType;
    public CustomParameters customParameters = new CustomParameters();

    public static TimeEvalArguments fromJson(String jsonString) {
        TimeEvalArguments tea = new Gson().fromJson(jsonString, TimeEvalArguments.class);
        if (tea.executionType == null || tea.dataInput == null || tea.dataOutput == null) {
            System.err.println("All required parameters (dataInput, dataOutput, executionType) must be specified!");
            System.exit(1);
        }
        if (tea.executionType.equals("train")) {
            System.out.println("Nothing to train, finished!");
            System.exit(0);
        }
        return tea;
    }

    public void overwriteParameters(GrammarVizAnomalyParameters params) {
        // check some parameter invariances
        if (this.customParameters.saxWindowSize < this.customParameters.saxPaaSize) {
            System.out.println("Warning: anomaly_window_size must be greater than or equal to paa_transform_size!" +
                    "Dynamically fixing this issue by setting anomaly_window_size to paa_transform_size (" +
                    this.customParameters.saxPaaSize + ")."
            );
            this.customParameters.saxWindowSize = this.customParameters.saxPaaSize;
        }
        params.IN_FILE = this.dataInput;
        params.OUT_FILE = "out"; // set the prefix to something so that the files will be created
        params.DISTANCE_FILENAME = this.dataOutput;
        params.SAX_WINDOW_SIZE = this.customParameters.saxWindowSize;
        params.SAX_PAA_SIZE = this.customParameters.saxPaaSize;
        params.SAX_ALPHABET_SIZE = this.customParameters.saxAlphabetSize;
        params.SAX_NORM_THRESHOLD = this.customParameters.saxNormThreshold;
        params.RANDOM_SEED = this.customParameters.randomState;
        params.DISCORDS_NUM = this.customParameters.nDiscords;
        params.OUTPUT_MODE = OutputMode.values()[this.customParameters.outputMode];
        params.MULTIVARIATE_STRATEGY = MultivariateStrategy.values()[this.customParameters.multivariateStrategy];
        params.SAX_NR_STRATEGY = this.customParameters.numerosityReduction ? NumerosityReductionStrategy.EXACT : NumerosityReductionStrategy.NONE;
    }

    static final class CustomParameters {
        @SerializedName("anomaly_window_size")
        public int saxWindowSize = 100;
        @SerializedName("paa_transform_size")
        public int saxPaaSize = 5;
        @SerializedName("alphabet_size")
        public int saxAlphabetSize = 6;
        @SerializedName("normalization_threshold")
        public double saxNormThreshold = 0.01;
        @SerializedName("random_state")
        public long randomState = 42;
        @SerializedName("n_discords")
        public int nDiscords = 10;
        @SerializedName("output_mode")
        public int outputMode = 2;
        @SerializedName("multi_strategy")
        public int multivariateStrategy = 1;
        @SerializedName("numerosity_reduction")
        public boolean numerosityReduction = true;

        @Override
        public String toString() {
            return "CustomParameters{" +
                    "SAX_WINDOW_SIZE=" + saxWindowSize +
                    ", SAX_PAA_SIZE=" + saxPaaSize +
                    ", SAX_ALPHABET_SIZE=" + saxAlphabetSize +
                    ", SAX_NORM_THRESHOLD=" + saxNormThreshold +
                    ", RANDOM_STATE=" + randomState +
                    ", DISCORDS_NUM=" + nDiscords +
                    '}';
        }
    }

    @Override
    public String toString() {
        return "TimeEvalArguments{" +
                "dataInput='" + dataInput + '\'' +
                ", dataOutput='" + dataOutput + '\'' +
                ", executionType='" + executionType + '\'' +
                ", customParameters=" + customParameters +
                '}';
    }
}
