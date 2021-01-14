package zero.deeplearning.ch06;

import org.apache.commons.math3.linear.RealMatrix;
import zero.deeplearning.common.MnistDataset;
import zero.deeplearning.network.MultiLayerNetExtend;
import zero.deeplearning.optimizer.Optimizer;
import zero.deeplearning.optimizer.SGD;

import java.io.IOException;
import java.util.HashMap;

import static zero.deeplearning.common.Utils.extractRow;
import static zero.deeplearning.common.Utils.randomChoice;

public class BatchNormTest {
    public static void main(String... args) throws IOException {
        HashMap<String, RealMatrix> dataset = MnistDataset.loadMinist(true, true);
        RealMatrix xTrain = extractRow(dataset.get("train_img"), 0, 999);
        RealMatrix tTrain = extractRow(dataset.get("train_label"), 0, 999);

        int maxEpochs = 20;
        int trainSize = xTrain.getRowDimension();
        int batchSize = 100;
        double learningRate = 0.01;

        for (int i = 0; i < 16; i++) {
            double w = Math.pow(10.0, -i * 4.0 / 15.0);
            System.out.println("============== " + (i + 1) + "/16" + " (w=" + w + ") ==============");

            MultiLayerNetExtend bnNetwork = new MultiLayerNetExtend(784, new int[]{100, 100, 100, 100, 100}, 10, "ReLU", String.valueOf(w), 0.0, false, 0.5, true);
            MultiLayerNetExtend network = new MultiLayerNetExtend(784, new int[]{100, 100, 100, 100, 100}, 10, "ReLU", String.valueOf(w), 0.0, false, 0.5, false);
            Optimizer optimizer = new SGD(learningRate);

            int iterPerEpoch = trainSize / batchSize;
            int epochCnt = 0;

            for (int j = 0; j < 1000000000; j++) {
                int[] batchMask = randomChoice(trainSize, batchSize);
                RealMatrix xBatch = extractRow(xTrain, batchMask);
                RealMatrix tBatch = extractRow(tTrain, batchMask);

                for (MultiLayerNetExtend networkTmp : new MultiLayerNetExtend[]{bnNetwork, network}) {
                    HashMap<String, RealMatrix> grads = networkTmp.gradient(xBatch, tBatch);
                    optimizer.update(networkTmp.params, grads);
                    networkTmp.update();
                }

                if (j % iterPerEpoch == 0) {
                    double trainAcc = network.accuracy(xTrain, tTrain);
                    double bnTrainAcc = bnNetwork.accuracy(xTrain, tTrain);

                    System.out.println("epoch:" + epochCnt + " | " + trainAcc + " - " + bnTrainAcc);

                    epochCnt += 1;
                    if (epochCnt >= maxEpochs) break;
                }
            }
        }
    }
}
