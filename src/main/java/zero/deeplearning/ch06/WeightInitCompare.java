package zero.deeplearning.ch06;

import org.apache.commons.math3.linear.RealMatrix;
import zero.deeplearning.common.MnistDataset;
import zero.deeplearning.network.MultiLayerNet;
import zero.deeplearning.optimizer.Optimizer;
import zero.deeplearning.optimizer.SGD;

import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import static zero.deeplearning.common.Utils.extractRow;
import static zero.deeplearning.common.Utils.randomChoice;

public class WeightInitCompare {
    public static void main(String... args) throws IOException {
        HashMap<String, RealMatrix> dataset = MnistDataset.loadMinist(true, true);
        RealMatrix xTrain = dataset.get("train_img");
        RealMatrix tTrain = dataset.get("train_label");

        int trainSize = xTrain.getRowDimension();
        int batchSize = 128;
        int maxIterations = 2000;

        Map<String, String> weightInitTypes = new LinkedHashMap<>();
        weightInitTypes.put("std=0.01", "0.01");
        weightInitTypes.put("Xavier", "sigmoid");
        weightInitTypes.put("He", "relu");
        Optimizer optimizer = new SGD(0.01);

        Map<String, MultiLayerNet> networks = new HashMap<>();
        for (String key : weightInitTypes.keySet()) {
            networks.put(key, new MultiLayerNet(784, new int[]{100, 100, 100, 100}, 10, "ReLU", weightInitTypes.get(key), 0.0));
        }

        for (int i = 0; i < maxIterations; i++) {
            int[] batchMask = randomChoice(trainSize, batchSize);
            RealMatrix xBatch = extractRow(xTrain, batchMask);
            RealMatrix tBatch = extractRow(tTrain, batchMask);

            for (String key : weightInitTypes.keySet()) {
                MultiLayerNet network = networks.get(key);
                HashMap<String, RealMatrix> grads = network.gradient(xBatch, tBatch);
                optimizer.update(network.params, grads);
                network.update();
            }

            if (i % 100 == 0) {
                System.out.println("===========" + "iteration:" + i + "===========");
                for (String key : weightInitTypes.keySet()) {
                    MultiLayerNet network = networks.get(key);
                    double loss = network.loss(xBatch, tBatch);
                    System.out.println(key + ":" + loss);
                }
            }
        }
    }
}
