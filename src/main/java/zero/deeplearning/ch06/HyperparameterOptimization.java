package zero.deeplearning.ch06;

import org.apache.commons.math3.linear.RealMatrix;
import zero.deeplearning.common.MnistDataset;
import zero.deeplearning.network.MultiLayerNet;
import zero.deeplearning.network.Network;
import zero.deeplearning.network.Trainer;

import java.io.IOException;
import java.util.*;

import static zero.deeplearning.common.Utils.*;

public class HyperparameterOptimization {

    public static void main(String... args) throws IOException {
        HashMap<String, RealMatrix> dataset = MnistDataset.loadMinist(true, true);
        RealMatrix xTrain = extractRow(dataset.get("train_img"), 0, 499);
        RealMatrix tTrain = extractRow(dataset.get("train_label"), 0, 499);
        RealMatrix xTest = dataset.get("test_img");
        RealMatrix tTest = dataset.get("test_label");

        double validationRate = 0.2;
        int validationNum = (int) (xTrain.getRowDimension() * validationRate);
        List<Integer> index = shuffleIndex(xTrain.getRowDimension());
        xTrain = shuffleDataset(xTrain, index);
        tTrain = shuffleDataset(tTrain, index);
        RealMatrix xVal = extractRow(xTrain, 0, validationNum - 1);
        RealMatrix tVal = extractRow(tTrain, 0, validationNum - 1);
        xTrain = extractRow(xTrain, validationNum, xTrain.getRowDimension() - 1);
        tTrain = extractRow(tTrain, validationNum, tTrain.getRowDimension() - 1);

        int optimizationTrial = 100;
        Map<String, List<Double>> resultsVal = new LinkedHashMap<>();
        Map<String, List<Double>> resultsTrain = new LinkedHashMap<>();
        Random rand = new Random();
        for (int i = 0; i < optimizationTrial; i++) {
            double weightDecay = Math.pow(10, -4 - 4 * rand.nextDouble());
            double lr = Math.pow(10, -2 - 4 * rand.nextDouble());

            Network network = new MultiLayerNet(784, new int[]{100, 100, 100, 100, 100, 100}, 10, "ReLU", "ReLU", weightDecay);
            Map<String, Double> optimizerParam = new HashMap<>();
            optimizerParam.put("lr", lr);
            Trainer trainer = new Trainer(network, xTrain, tTrain, xVal, tVal, 50, 100, "SGD", optimizerParam, 0, false);
            trainer.train();
            List<Double> valAccList = trainer.getTrainAccList();
            List<Double> trainAccList = trainer.getTrainAccList();
            System.out.println("val acc:" + valAccList.get(valAccList.size() - 1) + " | lr:" + lr + ", weight decay:" + weightDecay);
            String key = "lr:" + lr + ", weight decay:" + weightDecay;
            resultsVal.put(key, valAccList);
            resultsTrain.put(key, trainAccList);
        }

        System.out.println("=========== Hyper-Parameter Optimization Result ===========");
        Map<String, Double> rank = new HashMap<>();
        for (String key : resultsVal.keySet()) {
            rank.put(key, resultsVal.get(key).get(resultsVal.get(key).size() - 1));
        }
        List<Map.Entry<String, Double>> listEntries = new ArrayList<Map.Entry<String, Double>>(rank.entrySet());
        Collections.sort(listEntries, new Comparator<Map.Entry<String, Double>>() {
            public int compare(Map.Entry<String, Double> obj1, Map.Entry<String, Double> obj2) {
                return obj2.getValue().compareTo(obj1.getValue());
            }
        });
        int i = 0;
        for (Map.Entry<String, Double> entry : listEntries) {
            System.out.println("Best-" + (i + 1) + "(val acc:" + resultsVal.get(entry.getKey()).get(resultsVal.get(entry.getKey()).size() - 1) + ") | " + entry.getKey());
            i += 1;
            if(i >= 20) break;
        }
    }
}
