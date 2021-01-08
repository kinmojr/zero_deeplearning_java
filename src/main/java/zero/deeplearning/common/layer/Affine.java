package zero.deeplearning.common.layer;

import org.apache.commons.math3.linear.RealMatrix;

import static zero.deeplearning.common.Functions.*;

public class Affine extends Layer {
    RealMatrix w;
    RealMatrix b;
    RealMatrix x;
    RealMatrix dW;
    RealMatrix dB;

    public Affine(RealMatrix w, RealMatrix b) {
        this.w = w;
        this.b = b;
    }

    @Override
    public RealMatrix forward(RealMatrix x) {
        this.x = x;
        return addBias(dot(x, w), b);
    }

    @Override
    public RealMatrix backward(RealMatrix dout) {
        RealMatrix dx = dot(dout, t(w));
        dW = dot(t(x), dout);
        dB = sumCol(dout);
        return dx;
    }

    @Override
    public RealMatrix dw() {
        return dW;
    }

    @Override
    public RealMatrix db() {
        return dB;
    }

    @Override
    public void update(RealMatrix w, RealMatrix b){
        this.w = w;
        this.b = b;
    }
}
