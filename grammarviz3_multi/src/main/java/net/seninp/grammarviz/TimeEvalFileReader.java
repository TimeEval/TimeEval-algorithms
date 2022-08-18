package net.seninp.grammarviz;

import net.seninp.grammarviz.multivariate.timeseries.UnivariateTimeSeries;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public final class TimeEvalFileReader {
    static final String DEFAULT_CSV_DELIMITER = ",";

    private static final Logger LOGGER = LoggerFactory.getLogger(TimeEvalFileReader.class);

    private TimeEvalFileReader() {
    }

    static public List<UnivariateTimeSeries> readTS(String filename) throws IOException {
        return readTS(filename, DEFAULT_CSV_DELIMITER);
    }

    static public List<UnivariateTimeSeries> readTS(String filename, String delimiter) throws IOException {
        List<List<Double>> values = new ArrayList<>();
        int wronglyFormattedValues = 0;
        try (BufferedReader in = new BufferedReader(new FileReader(filename))) {
            String line;
            String header = in.readLine();
            int nCols = header.split(delimiter).length;
            for (int i = 0; i < nCols - 2; i++) {
                values.add(new ArrayList<>());
            }

            while ((line = in.readLine()) != null) {
                String[] rowValues = line.split(delimiter);

                //skip first (timestamp) and last (is_anomaly) columns
                for (int i = 1; i < nCols - 1; i++) {
                    String value = rowValues[i].trim();
                    try {
                        values.get(i - 1).add(Double.parseDouble(value));
                    } catch (NumberFormatException e) {
                        values.get(i - 1).add(Double.NaN);
                        wronglyFormattedValues++;
                    }
                }
            }
        } finally {
            if (wronglyFormattedValues > 0)
                LOGGER.warn("Warning: Read {} wrongly formatted values!", wronglyFormattedValues);
        }
        return values.stream()
                .map(dimension -> new UnivariateTimeSeries(dimension.stream().mapToDouble(i -> i).toArray()))
                .collect(Collectors.toList());
    }
}
