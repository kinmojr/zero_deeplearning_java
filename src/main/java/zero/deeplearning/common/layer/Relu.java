package zero.deeplearning.common.layer;

import static zero.deeplearning.common.Functions.*;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

public class Relu extends Layer {
    RealMatrix mask;

    @Override
    public RealMatrix forward(RealMatrix x) {
        double[][] tmp = x.getData();
        for (int i = 0; i < tmp.length; i++) {
            for (int j = 0; j < tmp[i].length; j++) {
                if (tmp[i][j] < 0.0) tmp[i][j] = 0.0;
            }
        }
        mask = MatrixUtils.createRealMatrix(tmp);
        return mult(x, mask);
    }

    @Override
    public RealMatrix backward(RealMatrix dout) {
        return mult(dout, mask);
    }
}
