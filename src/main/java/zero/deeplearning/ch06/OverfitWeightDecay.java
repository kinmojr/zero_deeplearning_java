package zero.deeplearning.ch06;

import org.apache.commons.math3.linear.RealMatrix;
import zero.deeplearning.common.MnistDataset;
import zero.deeplearning.network.MultiLayerNet;
import zero.deeplearning.optimizer.Optimizer;
import zero.deeplearning.optimizer.SGD;

import java.io.IOException;
import java.util.HashMap;

import static zero.deeplearning.common.Utils.extractRow;
import static zero.deeplearning.common.Utils.randomChoice;

public class OverfitWeightDecay {
    public static void main(String... args) throws IOException {
        HashMap<String, RealMatrix> dataset = MnistDataset.loadMinist(true, true);
        RealMatrix xTrain = extractRow(dataset.get("train_img"), 0, 299);
        RealMatrix tTrain = extractRow(dataset.get("train_label"), 0, 299);
        RealMatrix xTest = dataset.get("test_img");
        RealMatrix tTest = dataset.get("test_label");

        double weightDecayLambda = 0.1;

        MultiLayerNet network = new MultiLayerNet(784, new int[]{100, 100, 100, 100, 100}, 10, "ReLU", "ReLU", weightDecayLambda);
        Optimizer optimizer = new SGD(0.01);

        int maxEpochs = 201;
        int trainSize = xTrain.getRowDimension();
        int batchSize = 100;

        int iterPerEpoch = trainSize / batchSize;
        int epochCnt = 0;

        for (int i = 0; i < 1000000000; i++) {
            int[] batchMask = randomChoice(trainSize, batchSize);
            RealMatrix xBatch = extractRow(xTrain, batchMask);
            RealMatrix tBatch = extractRow(tTrain, batchMask);

            HashMap<String, RealMatrix> grads = network.gradient(xBatch, tBatch);
            optimizer.update(network.params, grads);
            network.update();

            if (i % iterPerEpoch == 0) {
                double trainAcc = network.accuracy(xTrain, tTrain);
                double testAcc = network.accuracy(xTest, tTest);

                System.out.println("epoch:" + epochCnt + ", train acc:" + trainAcc + ", test acc:" + testAcc);

                epochCnt += 1;
                if (epochCnt >= maxEpochs) break;
            }
        }
    }
}
