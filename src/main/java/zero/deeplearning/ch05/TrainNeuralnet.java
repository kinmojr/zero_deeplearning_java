package zero.deeplearning.ch05;

import org.apache.commons.math3.linear.RealMatrix;
import zero.deeplearning.common.MnistDataset;

import java.io.IOException;
import java.util.HashMap;

import static zero.deeplearning.common.Functions.mult;
import static zero.deeplearning.common.Functions.sub;
import static zero.deeplearning.common.Utils.extractRow;
import static zero.deeplearning.common.Utils.randomChoice;

public class TrainNeuralnet {
    public static void main(String... args) throws IOException {
        HashMap<String, RealMatrix> dataset = MnistDataset.loadMinist(true, true);
        RealMatrix xTrain = dataset.get("train_img");
        RealMatrix tTrain = dataset.get("train_label");
        RealMatrix xTest = dataset.get("test_img");
        RealMatrix tTest = dataset.get("test_label");

        int itersNum = 10000;
        int trainSize = xTrain.getRowDimension();
        int batchSize = 100;
        double learningRate = 0.1;
        TwoLayerNet network = new TwoLayerNet(784, 50, 10, 0.01);

        int iterPerEpoch = trainSize / batchSize;

        for (int i = 0; i < itersNum; i++) {
            int[] indexes = randomChoice(trainSize, batchSize);
            RealMatrix xBatch = extractRow(xTrain, indexes);
            RealMatrix tBatch = extractRow(tTrain, indexes);

            HashMap<String, RealMatrix> grads = network.gradient(xBatch, tBatch);

            for (String key : grads.keySet()) {
                network.params.put(key, sub(network.params.get(key), mult(grads.get(key), learningRate)));
            }
            network.update();

            if (i % iterPerEpoch == 0) {
                double trainAcc = network.accuracy(xTrain, tTrain);
                double testAcc = network.accuracy(xTest, tTest);
                System.out.println("train acc, test acc | " + trainAcc + ", " + testAcc);
            }
        }
    }
}
