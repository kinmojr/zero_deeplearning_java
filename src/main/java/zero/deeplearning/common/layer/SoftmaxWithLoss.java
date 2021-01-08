package zero.deeplearning.common.layer;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import static zero.deeplearning.common.Functions.*;

public class SoftmaxWithLoss extends Layer {
    private RealMatrix y;
    private RealMatrix t;

    @Override
    public double forward(RealMatrix x, RealMatrix t) {
        this.t = t;
        y = MatrixUtils.createRealMatrix(softmax(x.getData()));
        return crossEntropyError(y, t);
    }

    @Override
    public RealMatrix backward(RealMatrix dout) {
        return div(sub(y, t), t.getRowDimension());
    }
}
