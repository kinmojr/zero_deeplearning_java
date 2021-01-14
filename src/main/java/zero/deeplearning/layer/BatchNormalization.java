package zero.deeplearning.layer;

import org.apache.commons.math3.linear.RealMatrix;

import static zero.deeplearning.common.Functions.*;
import static zero.deeplearning.common.Utils.createMatrix;

public class BatchNormalization extends Layer {
    private RealMatrix gamma;
    private RealMatrix beta;
    private double momentum;
    private RealMatrix runningMean;
    private RealMatrix runningVar;
    private int batchSize;
    private RealMatrix xc;
    private RealMatrix xn;
    private RealMatrix std;
    private RealMatrix dgamma;
    private RealMatrix dbeta;

    public BatchNormalization(RealMatrix gamma, RealMatrix beta, double momentum, RealMatrix runningMean, RealMatrix runningVar) {
        this.gamma = gamma;
        this.beta = beta;
        this.momentum = momentum;
        this.runningMean = runningMean;
        this.runningVar = runningVar;
    }

    public BatchNormalization(RealMatrix gamma, RealMatrix beta) {
        this(gamma, beta, 0.9, null, null);
    }

    @Override
    public RealMatrix forward(RealMatrix x, boolean trainFlag) {
        if (runningMean == null) {
            runningMean = createMatrix(1, x.getColumnDimension());
            runningVar = createMatrix(1, x.getColumnDimension());
        }

        if (trainFlag) {
            RealMatrix mu = div(sumCol(x), x.getRowDimension());
            xc = sub(x, mu);
            RealMatrix var = div(sumCol(pow(xc, 2.0)), xc.getRowDimension());
            std = sqrt(add(var, 10e-7));
            xn = div(xc, std);

            batchSize = x.getRowDimension();
            runningMean = add(mult(runningMean, momentum), mult(mu, 1 - momentum));
            runningVar = add(mult(runningVar, momentum), mult(var, 1 - momentum));
        } else {
            xc = sub(x, runningMean);
            xn = div(xc, sqrt(add(runningVar, 10e-7)));
        }

        return add(mult(xn, gamma), beta);
    }

    @Override
    public RealMatrix backward(RealMatrix dout) {
        dbeta = sumCol(dout);
        dgamma = sumCol(mult(xn, dout));
        RealMatrix dxn = mult(gamma, dout);
        RealMatrix dxc = div(dxn, std);
        RealMatrix dstd = mult(sumCol(div(mult(dxn, xc), mult(std, std))), -1.0);
        RealMatrix dvar = mult(div(dstd, std), 0.5);
        dxc = add(dxc, mult(mult(xc, dvar), 2.0 / batchSize));
        RealMatrix dmu = sumCol(dxc);

        return sub(dxc, div(dmu, batchSize));
    }

    @Override
    public boolean hasTrainFlag() {
        return true;
    }

    @Override
    public RealMatrix dgamma() {
        return dgamma;
    }

    @Override
    public RealMatrix dbeta() {
        return dbeta;
    }

    @Override
    public void update(RealMatrix gamma, RealMatrix beta) {
        this.gamma = gamma;
        this.beta = beta;
    }
}
