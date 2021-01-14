package zero.deeplearning.network;


import org.apache.commons.math3.linear.RealMatrix;
import zero.deeplearning.optimizer.Optimizer;

import java.util.HashMap;
import java.util.Map;

import static zero.deeplearning.common.Utils.*;

public class Trainer {
    private Network network;
    private RealMatrix xTrain;
    private RealMatrix tTrain;
    private RealMatrix xTest;
    private RealMatrix tTest;
    private int epochs;
    private int batchSize;
    private Optimizer optimizer;
    private Map<String, Double> optimizerParam;
    private int evaluateSampleNumPerEpoch;
    private boolean verbose;
    private int trainSize;
    private int iterPerEpoch;
    private int maxIter;
    private int currentIter;
    private int currentEpoch;

    public Trainer(Network network, RealMatrix xTrain, RealMatrix tTrain, RealMatrix xTest, RealMatrix tTest, int epochs, int miniBatchSize, String optimizerName, Map<String, Double> optimizerParam, int evaluateSampleNumPerEpoch, boolean verbose) {
        this.network = network;
        this.verbose = verbose;
        this.xTrain = xTrain;
        this.tTrain = tTrain;
        this.xTest = xTest;
        this.tTest = tTest;
        this.epochs = epochs;
        this.batchSize = miniBatchSize;
        this.optimizerParam = optimizerParam;
        this.evaluateSampleNumPerEpoch = evaluateSampleNumPerEpoch;

        optimizer = createOptimizer(optimizerName, optimizerParam);

        trainSize = xTrain.getRowDimension();
        iterPerEpoch = trainSize / miniBatchSize;
        maxIter = epochs * iterPerEpoch;
        currentIter = 0;
        currentEpoch = 0;
    }

    private void trainStep() {
        int[] batchMask = randomChoice(trainSize, batchSize);
        RealMatrix xBatch = extractRow(xTrain, batchMask);
        RealMatrix tBatch = extractRow(tTrain, batchMask);

        HashMap<String, RealMatrix> grads = network.gradient(xBatch, tBatch);
        optimizer.update(network.getParams(), grads);
        network.update();

        double loss = network.loss(xBatch, tBatch, false);
        if (verbose) System.out.println("train loss:" + loss);

        if (currentIter % iterPerEpoch == 0) {
            currentEpoch += 1;

            RealMatrix xTrainSample = xTrain.copy();
            RealMatrix tTrainSample = tTrain.copy();
            RealMatrix xTestSample = xTest.copy();
            RealMatrix tTestSample = tTest.copy();
            if (evaluateSampleNumPerEpoch > 0) {
                int t = evaluateSampleNumPerEpoch;
                xTrainSample = extractRow(xTrainSample, 0, t - 1);
                tTrainSample = extractRow(tTrainSample, 0, t - 1);
                xTestSample = extractRow(xTestSample, 0, t - 1);
                tTestSample = extractRow(tTestSample, 0, t - 1);
            }

            double trainAcc = network.accuracy(xTrainSample, tTrainSample);
            double testAcc = network.accuracy(xTestSample, tTestSample);
            if (verbose) {
                System.out.println("=== epoch:" + currentEpoch + ", train acc:" + trainAcc + ", test acc:" + testAcc + " ===");
            }
        }
        currentIter += 1;
    }

    public void train() {
        for (int i = 0; i < maxIter; i++) {
            trainStep();
        }

        double testAcc = network.accuracy(xTest, tTest);

        if (verbose) {
            System.out.println("=============== Final Test Accuracy ===============");
            System.out.println("test acc:" + testAcc);
        }
    }
}
