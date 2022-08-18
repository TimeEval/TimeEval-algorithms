package net.seninp.grammarviz.multivariate.timeseries;

import net.seninp.grammarviz.TimeEvalFileReader;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;

public class MultivariateTimeSeries implements Iterable<UnivariateTimeSeries> {
    private final List<UnivariateTimeSeries> ts;

    public MultivariateTimeSeries(String fileName) throws IOException {
        this.ts = TimeEvalFileReader.readTS(fileName);
    }

    public int getLength() {
        return !ts.isEmpty() ? ts.get(0).getLength() : 0;
    }

    public int getWidth() {
        return ts.size();
    }

    public boolean isUnivariate() {
        return ts.size() == 1;
    }

    public double get(int dimension, int position) {
        return ts.get(dimension).get(position);
    }

    public UnivariateTimeSeries get(int dimension) {
        return ts.get(dimension);
    }

    public void remove(int i) {
        ts.remove(i);
    }

    public String toString() {
        return "TimeSeries with " + getWidth() + " dimensions and a length of " + getLength() + " data points";
    }

    @Override
    public Iterator<UnivariateTimeSeries> iterator() {
        return ts.iterator();
    }
}
