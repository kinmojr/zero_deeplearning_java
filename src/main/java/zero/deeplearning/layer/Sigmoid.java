package zero.deeplearning.layer;

import org.apache.commons.math3.linear.RealMatrix;

import static zero.deeplearning.common.Functions.sigmoid;
import static zero.deeplearning.common.Utils.createMatrix;

public class Sigmoid extends Layer {
    RealMatrix out;

    @Override
    public RealMatrix forward(RealMatrix x) {
        this.out = sigmoid(x);
        return out;
    }

    @Override
    public RealMatrix backward(RealMatrix dout) {
        RealMatrix dx = createMatrix(dout.getRowDimension(), dout.getColumnDimension());
        for (int i = 0; i < dout.getRowDimension(); i++) {
            for (int j = 0; j < dout.getColumnDimension(); j++) {
                dx.setEntry(i, j, dout.getEntry(i, j) * (1.0 - out.getEntry(i, j)) * out.getEntry(i, j));
            }
        }
        return dx;
    }
}
