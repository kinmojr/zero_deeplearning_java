package zero.deeplearning.network;

import org.apache.commons.math3.linear.RealMatrix;

import java.util.LinkedHashMap;
import java.util.Map;

public abstract class Network {
    public RealMatrix predict(RealMatrix x) {
        throw new RuntimeException("This method should not be called.");
    }

    public RealMatrix predict(RealMatrix x, boolean trainFlag) {
        throw new RuntimeException("This method should not be called.");
    }

    public double loss(RealMatrix x, RealMatrix t) {
        throw new RuntimeException("This method should not be called.");
    }

    public double loss(RealMatrix x, RealMatrix t, boolean trainFlag) {
        throw new RuntimeException("This method should not be called.");
    }

    public double accuracy(RealMatrix x, RealMatrix t) {
        throw new RuntimeException("This method should not be called.");
    }

    public LinkedHashMap<String, RealMatrix> gradient(RealMatrix x, RealMatrix t) {
        throw new RuntimeException("This method should not be called.");
    }

    public void update() {
        throw new RuntimeException("This method should not be called.");
    }

    public Map<String, RealMatrix> getParams() {
        throw new RuntimeException("This method should not be called.");
    }
}
