package zero.deeplearning.layer;

import org.apache.commons.math3.linear.RealMatrix;

import java.util.Random;

import static zero.deeplearning.common.Functions.mult;
import static zero.deeplearning.common.Utils.createMatrix;

public class Dropout extends Layer {
    private double dropoutRatio;
    private RealMatrix mask;

    public Dropout(double dropoutRatio) {
        this.dropoutRatio = dropoutRatio;
    }

    public Dropout() {
        this.dropoutRatio = 0.5;
    }

    @Override
    public RealMatrix forward(RealMatrix x, boolean trainFlag) {
        if (trainFlag) {
            mask = createMatrix(x.getRowDimension(), x.getColumnDimension());
            Random rand = new Random();
            for (int i = 0; i < x.getRowDimension(); i++) {
                for (int j = 0; j < x.getColumnDimension(); j++) {
                    if (rand.nextDouble() > dropoutRatio)
                        mask.setEntry(i, j, 1.0);
                }
            }
            return mult(x, mask);
        } else {
            return mult(x, 1 - dropoutRatio);
        }
    }

    @Override
    public RealMatrix backward(RealMatrix dout) {
        return mult(dout, mask);
    }

    @Override
    public boolean hasTrainFlag() {
        return true;
    }
}
