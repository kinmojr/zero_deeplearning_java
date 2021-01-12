package zero.deeplearning.ch06;

import org.apache.commons.math3.linear.RealMatrix;

import static zero.deeplearning.common.Functions.*;
import static zero.deeplearning.common.Utils.*;

import zero.deeplearning.common.layer.Layer;
import zero.deeplearning.common.layer.Affine;
import zero.deeplearning.common.layer.Relu;
import zero.deeplearning.common.layer.Sigmoid;
import zero.deeplearning.common.layer.SoftmaxWithLoss;

import java.util.*;

public class MultiLayerNet {
    private int inputSize;
    private int[] hiddenSizeList;
    private int outputSize;
    private double weightDecayLambda;
    private int hiddenLayerNum;
    public Map<String, RealMatrix> params = new LinkedHashMap<>();
    private Map<String, Layer> layers = new LinkedHashMap<>();
    private Layer lastLayer;

    public MultiLayerNet(int inputSize, int[] hiddenSizeList, int outputSize, String activation, String weightInitStd, double weightDecayLambda) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.hiddenSizeList = hiddenSizeList;
        this.hiddenLayerNum = hiddenSizeList.length;
        this.weightDecayLambda = weightDecayLambda;

        initWeight(weightInitStd);

        int idx = 1;
        while (idx < hiddenLayerNum + 1) {
            layers.put("Affine" + idx, new Affine(params.get("W" + idx), params.get("b" + idx)));
            if ("sigmoid".equals(activation.toLowerCase())) {
                layers.put("Activation_function" + idx, new Sigmoid());
            } else if ("relu".equals(activation.toLowerCase())) {
                layers.put("Activation_function" + idx, new Relu());
            }
            idx++;
        }
        idx = hiddenLayerNum + 1;
        layers.put("Affine" + idx, new Affine(params.get("W" + idx), params.get("b" + idx)));

        lastLayer = new SoftmaxWithLoss();
    }

    private void initWeight(String weightInitStd) {
        int[] allSizeList = new int[hiddenLayerNum + 2];
        allSizeList[0] = inputSize;
        for (int i = 0; i < hiddenLayerNum; i++) {
            allSizeList[i + 1] = hiddenSizeList[i];
        }
        allSizeList[hiddenLayerNum + 1] = outputSize;
        for (int idx = 1; idx < allSizeList.length; idx++) {
            double scale;
            if ("relu".equals(weightInitStd.toLowerCase()) || "he".equals(weightInitStd.toLowerCase())) {
                scale = Math.sqrt(2.0 / allSizeList[idx - 1]);
            } else if ("sigmoid".equals(weightInitStd.toLowerCase()) || "xavier".equals(weightInitStd.toLowerCase())) {
                scale = Math.sqrt(1.0 / allSizeList[idx - 1]);
            } else {
                scale = Double.valueOf(weightInitStd);
            }
            params.put("W" + idx, mult(randomMatrix(allSizeList[idx - 1], allSizeList[idx]), scale));
            params.put("b" + idx, createMatrix(1, allSizeList[idx]));
        }
    }

    public RealMatrix predict(RealMatrix x) {
        for (Layer layer : layers.values())
            x = layer.forward(x);

        return x;
    }

    public double loss(RealMatrix x, RealMatrix t) {
        RealMatrix y = predict(x);

        double weightDecay = 0.0;
        for (int idx = 1; idx < hiddenLayerNum + 2; idx++) {
            RealMatrix W = params.get("W" + idx);
            weightDecay += 0.5 * weightDecayLambda * l2norm(W);
        }

        return lastLayer.forward(y, t) + weightDecay;
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

    public LinkedHashMap<String, RealMatrix> gradient(RealMatrix x, RealMatrix t) {
        loss(x, t);

        RealMatrix dout = createMatrix(1, 1, 1.0);
        dout = lastLayer.backward(dout);

        List<String> keyList = new ArrayList<>(layers.keySet());
        Collections.reverse(keyList);
        for (String key : keyList)
            dout = layers.get(key).backward(dout);

        LinkedHashMap<String, RealMatrix> grads = new LinkedHashMap<>();
        for (int idx = 1; idx < hiddenLayerNum + 2; idx++) {
            grads.put("W" + idx, add(layers.get("Affine" + idx).dw(), mult(layers.get("Affine" + idx).w(), weightDecayLambda)));
            grads.put("b" + idx, layers.get("Affine" + idx).db());
        }

        return grads;
    }

    public void update() {
        for (String key : layers.keySet()) {
            if (key.length() > 6 && "Affine".equals(key.substring(0, 6))) {
                int index = Integer.parseInt(key.substring(6));
                layers.get(key).update(params.get("W" + index), params.get("b" + index));
            }
        }
    }
}
