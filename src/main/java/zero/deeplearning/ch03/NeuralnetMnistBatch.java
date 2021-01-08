package zero.deeplearning.ch03;

import zero.deeplearning.common.MNIST;

import static zero.deeplearning.common.Utils.*;
import static zero.deeplearning.common.Functions.*;

import org.apache.commons.math3.linear.RealMatrix;

import java.io.IOException;
import java.util.HashMap;

public class NeuralnetMnistBatch {
    private RealMatrix w1, w2, w3, b1, b2, b3;

    private NeuralnetMnistBatch() throws IOException {
        w1 = readWeights("mnist/w1.tsv");
        w2 = readWeights("mnist/w2.tsv");
        w3 = readWeights("mnist/w3.tsv");
        b1 = readWeights("mnist/b1.tsv");
        b2 = readWeights("mnist/b2.tsv");
        b3 = readWeights("mnist/b3.tsv");
    }

    private RealMatrix predict(RealMatrix x) {
        RealMatrix a1 = addBias(dot(x, w1), b1);
        RealMatrix z1 = sigmoid(a1);
        RealMatrix a2 = addBias(dot(z1, w2), b2);
        RealMatrix z2 = sigmoid(a2);
        RealMatrix a3 = addBias(dot(z2, w3), b3);
        return softmax(a3);
    }

    public static void main(String... args) throws IOException {
        HashMap<String, RealMatrix> dataset = MNIST.loadMinist(true, false);
        RealMatrix x = dataset.get("test_img");
        RealMatrix t = dataset.get("test_label");
        NeuralnetMnistBatch network = new NeuralnetMnistBatch();

        int batchSize = 100;
        int accuracyCnt = 0;
        for (int i = 0; i < x.getRowDimension(); i += batchSize) {
            RealMatrix xBatch = extractRow(x, i, i + batchSize - 1);
            RealMatrix yBatch = network.predict(xBatch);
            int[] p = argmax(yBatch.getData());
            for (int j = 0; j < p.length; j++) {
                if (p[j] == t.getEntry(i + j, 0)) {
                    accuracyCnt++;
                }
            }
        }
        System.out.println("Accuracy:" + (float) accuracyCnt / (float) x.getRowDimension());
    }
}
