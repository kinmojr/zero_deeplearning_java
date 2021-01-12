package zero.deeplearning.ch04;

import org.apache.commons.math3.linear.RealMatrix;

import java.util.HashMap;
import java.util.Map;

import static zero.deeplearning.common.Functions.*;
import static zero.deeplearning.common.Utils.*;

public class TwoLayerNet {
    public Map<String, RealMatrix> params;

    public TwoLayerNet(int inputSize, int hiddenSize, int outputSize, double weightInitStd) {
        params = new HashMap<>();
        params.put("w1", initWeight(inputSize, hiddenSize, weightInitStd));
        params.put("b1", initBias(hiddenSize));
        params.put("w2", initWeight(hiddenSize, outputSize, weightInitStd));
        params.put("b2", initBias(outputSize));
    }

    private RealMatrix predict(RealMatrix x) {
        RealMatrix a1 = addBias(dot(x, params.get("w1")), params.get("b1"));
        RealMatrix z1 = sigmoid(a1);
        RealMatrix a2 = addBias(dot(z1, params.get("w2")), params.get("b2"));
        return softmax(a2);
    }

    public double loss(RealMatrix x, RealMatrix t) {
        RealMatrix y = predict(x);
        return crossEntropyError(y, t);
    }

    public double accuracy(RealMatrix x, RealMatrix t) {
        int accuracyCnt = 0;
        int[] y = argmax(predict(x).getData());
        for (int i = 0; i < y.length; i++) {
            if (t.getEntry(i, y[i]) == 1.0) {
                accuracyCnt++;
            }
        }
        return (float) accuracyCnt / (float) x.getRowDimension();
    }

    public HashMap<String, RealMatrix> numericalGradient(RealMatrix x, RealMatrix t) {
        HashMap<String, RealMatrix> grads = new HashMap<>();
        for (String key : params.keySet())
            grads.put(key, numericalGradient(params.get(key), x, t));
        return grads;
    }

    public RealMatrix numericalGradient(RealMatrix w, RealMatrix x, RealMatrix t) {
        double[][] grads = new double[w.getRowDimension()][w.getColumnDimension()];
        double diff = Math.pow(10, -4);
        for (int i = 0; i < grads.length; i++) {
            for (int j = 0; j < grads[i].length; j++) {
                double tmp = w.getEntry(i, j);

                w.setEntry(i, j, tmp + diff);
                double value1 = loss(x, t);

                w.setEntry(i, j, tmp - diff);
                double value2 = loss(x, t);

                grads[i][j] = (value1 - value2) / (2 * diff);
                w.setEntry(i, j, tmp);
            }
        }
        return createMatrix(grads);
    }

    public HashMap<String, RealMatrix> gradient(RealMatrix x, RealMatrix t) {
        HashMap<String, RealMatrix> grads = new HashMap<>();

        int batchNum = x.getRowDimension();

        RealMatrix a1 = addBias(dot(x, params.get("w1")), params.get("b1"));
        RealMatrix z1 = sigmoid(a1);
        RealMatrix a2 = addBias(dot(z1, params.get("w2")), params.get("b2"));
        RealMatrix y = softmax(a2);

        RealMatrix dy = div(sub(y, t), batchNum);
        grads.put("w2", dot(t(z1), dy));
        grads.put("b2", sumCol(dy));

        RealMatrix dz1 = dot(dy, t(params.get("w2")));
        RealMatrix da1 = mult(sigmoidGrad(a1), dz1);
        grads.put("w1", dot(t(x), da1));
        grads.put("b1", sumCol(da1));

        return grads;
    }
}
