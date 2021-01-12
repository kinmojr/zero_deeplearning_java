package zero.deeplearning.ch06;

import org.apache.commons.math3.linear.RealMatrix;
import zero.deeplearning.common.MNIST;
import zero.deeplearning.common.optimizer.*;

import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import static zero.deeplearning.common.Utils.*;

public class OptimizerCompareMnist {
    public static void main(String... args) throws IOException {
        HashMap<String, RealMatrix> dataset = MNIST.loadMinist(true, true);
        RealMatrix xTrain = dataset.get("train_img");
        RealMatrix tTrain = dataset.get("train_label");

        int trainSize = xTrain.getRowDimension();
        int batchSize = 128;
        int maxIterations = 2000;

        Map<String, Optimizer> optimizers = new LinkedHashMap<>();
        optimizers.put("SGD", new SGD());
        optimizers.put("Momentum", new Momentum());
        optimizers.put("AdaGrad", new AdaGrad());
        optimizers.put("Adam", new Adam());

        Map<String, MultiLayerNet> networks = new HashMap<>();
        for (String key : optimizers.keySet()) {
            networks.put(key, new MultiLayerNet(784, new int[]{100, 100, 100, 100}, 10, "ReLU", "ReLU", 0.0));
        }

        for (int i = 0; i < maxIterations; i++) {
            int[] batchMask = randomChoice(trainSize, batchSize);
            RealMatrix xBatch = extractRow(xTrain, batchMask);
            RealMatrix tBatch = extractRow(tTrain, batchMask);

            for (String key : optimizers.keySet()) {
                MultiLayerNet network = networks.get(key);
                HashMap<String, RealMatrix> grads = network.gradient(xBatch, tBatch);
                optimizers.get(key).update(network.params, grads);
                network.update();
            }

            if (i % 100 == 0) {
                System.out.println("===========" + "iteration:" + i + "===========");
                for (String key : optimizers.keySet()) {
                    MultiLayerNet network = networks.get(key);
                    double loss = network.loss(xBatch, tBatch);
                    System.out.println(key + ":" + loss);
                }
            }
        }
    }
}
