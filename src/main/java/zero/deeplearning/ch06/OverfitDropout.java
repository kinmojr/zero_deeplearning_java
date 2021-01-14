package zero.deeplearning.ch06;

import org.apache.commons.math3.linear.RealMatrix;
import zero.deeplearning.common.MnistDataset;
import zero.deeplearning.network.MultiLayerNetExtend;
import zero.deeplearning.network.Trainer;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static zero.deeplearning.common.Utils.extractRow;

public class OverfitDropout {
    public static void main(String... args) throws IOException {
        HashMap<String, RealMatrix> dataset = MnistDataset.loadMinist(true, true);
        RealMatrix xTrain = extractRow(dataset.get("train_img"), 0, 299);
        RealMatrix tTrain = extractRow(dataset.get("train_label"), 0, 299);
        RealMatrix xTest = dataset.get("test_img");
        RealMatrix tTest = dataset.get("test_label");

        boolean useDropout = true;
        double dropoutRate = 0.2;

        MultiLayerNetExtend network = new MultiLayerNetExtend(784, new int[]{100, 100, 100, 100, 100, 100}, 10, "ReLU", "ReLU", 0, useDropout, dropoutRate, false);
        Map<String, Double> optimizerParam = new HashMap<>();
        optimizerParam.put("lr", 0.01);
        Trainer trainer = new Trainer(network, xTrain, tTrain, xTest, tTest, 301, 100, "SGD", optimizerParam, 0, true);
        trainer.train();
    }
}
