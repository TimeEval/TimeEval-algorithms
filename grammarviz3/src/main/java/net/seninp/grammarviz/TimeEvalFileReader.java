package net.seninp.grammarviz;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public final class TimeEvalFileReader {
    static final String DEFAULT_CSV_DELIMITER = ",";

    private static final Logger LOGGER = LoggerFactory.getLogger(TimeEvalFileReader.class);

    private TimeEvalFileReader() {
    }

    static public double[] readTS(String filename, int useColumn, int skipCols) throws IOException {
        return readTS(filename, DEFAULT_CSV_DELIMITER, useColumn, skipCols);
    }

    static public double[] readTS(String filename, String delimiter) throws IOException {
        return readTS(filename, delimiter, 0, 0);
    }

    static public double[] readTS(String filename, String delimiter, int useColumn, int skipCols) throws IOException {
        List<Double> values = new ArrayList<>();
        int wronglyFormattedValues = 0;
        try (BufferedReader in = new BufferedReader(new FileReader(filename))) {
            String line;
            for(int i = skipCols; i > 0; i--) in.readLine();
            while ((line = in.readLine()) != null) {
                String[] cells = line.split(delimiter);
                int maxColumnIndex = cells.length;
                if(useColumn > maxColumnIndex - 1) {
                    LOGGER.warn(
                        "Warning: Selected column index {} is out of bounds (max index = {})! " +
                        "Using last channel!", useColumn, maxColumnIndex
                    );
                    useColumn = maxColumnIndex;
                }
                String value = cells[useColumn].trim();
                try {
                    values.add(Double.valueOf(value));
                } catch (NumberFormatException e) {
                    values.add(Double.NaN);
                    wronglyFormattedValues++;
                }
            }
        } finally {
            if(wronglyFormattedValues > 0)
                LOGGER.warn("Warning: Read {} wrongly formatted values!", wronglyFormattedValues);
        }
        double[] series = new double[values.size()];
        for (int i = 0; i < values.size(); i++) {
            series[i] = values.get(i);
        }
        return series;
    }
}
