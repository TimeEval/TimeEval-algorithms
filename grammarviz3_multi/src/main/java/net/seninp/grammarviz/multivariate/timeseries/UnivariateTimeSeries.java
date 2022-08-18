package net.seninp.grammarviz.multivariate.timeseries;

import java.util.Arrays;
import java.util.Iterator;

public class UnivariateTimeSeries implements Iterable<Double> {
    private final double[] ts;

    public UnivariateTimeSeries(double[] ts) {
        this.ts = ts;
    }

    public int getLength() {
        return ts.length;
    }

    public double get(int position) {
        return ts[position];
    }

    public double[] getRawData() {
        return ts;
    }

    @Override
    public Iterator<Double> iterator() {
        return Arrays.stream(ts).iterator();
    }
}
