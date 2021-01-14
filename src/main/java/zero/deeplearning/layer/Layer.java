package zero.deeplearning.layer;

import org.apache.commons.math3.linear.RealMatrix;

public abstract class Layer {
    public RealMatrix forward(RealMatrix x) {
        throw new RuntimeException("This method should not be called.");
    }

    public RealMatrix forward(RealMatrix x, boolean trainFlag) {
        throw new RuntimeException("This method should not be called.");
    }

    public double forward(RealMatrix x, RealMatrix t) {
        throw new RuntimeException("This method should not be called.");
    }

    public RealMatrix backward(RealMatrix dout) {
        throw new RuntimeException("This method should not be called.");
    }

    public RealMatrix dw() {
        throw new RuntimeException("This method should not be called.");
    }

    public RealMatrix db() {
        throw new RuntimeException("This method should not be called.");
    }

    public RealMatrix w() {
        throw new RuntimeException("This method should not be called.");
    }

    public RealMatrix b() {
        throw new RuntimeException("This method should not be called.");
    }

    public RealMatrix dgamma() {
        throw new RuntimeException("This method should not be called.");
    }

    public RealMatrix dbeta() {
        throw new RuntimeException("This method should not be called.");
    }

    public void update(RealMatrix w, RealMatrix b) {
        throw new RuntimeException("This method should not be called.");
    }

    public boolean hasTrainFlag() {
        return false;
    }
}
