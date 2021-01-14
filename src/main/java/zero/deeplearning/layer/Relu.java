package zero.deeplearning.layer;

import org.apache.commons.math3.linear.RealMatrix;

import static zero.deeplearning.common.Functions.mult;
import static zero.deeplearning.common.Utils.createMatrix;

public class Relu extends Layer {
    RealMatrix mask;

    @Override
    public RealMatrix forward(RealMatrix x) {
        mask = createMatrix(x.getRowDimension(), x.getColumnDimension(), 1.0);
        for (int i = 0; i < x.getRowDimension(); i++) {
            for (int j = 0; j < x.getColumnDimension(); j++) {
                if (x.getEntry(i, j) <= 0.0) mask.setEntry(i, j, 0.0);
            }
        }
        return mult(x, mask);
    }

    @Override
    public RealMatrix backward(RealMatrix dout) {
        return mult(dout, mask);
    }
}
