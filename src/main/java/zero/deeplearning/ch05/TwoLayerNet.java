package zero.deeplearning.ch05;

import org.apache.commons.math3.linear.RealMatrix;
import zero.deeplearning.common.layer.Affine;
import zero.deeplearning.common.layer.Layer;
import zero.deeplearning.common.layer.Relu;
import zero.deeplearning.common.layer.SoftmaxWithLoss;

import java.util.*;

import static zero.deeplearning.common.Utils.*;

public class TwoLayerNet {
    public Map<String, RealMatrix> params;
    public Map<String, Layer> layers;
    public Layer lastLayer;

    public TwoLayerNet(int inputSize, int hiddenSize, int outputSize, double weightInitStd) {
        params = new HashMap<>();
        params.put("w1", initWeight(inputSize, hiddenSize, weightInitStd));
        params.put("b1", initBias(hiddenSize));
        params.put("w2", initWeight(hiddenSize, outputSize, weightInitStd));
        params.put("b2", initBias(outputSize));

        layers = new LinkedHashMap<>();
        layers.put("Affine1", new Affine(params.get("w1"), params.get("b1")));
        layers.put("Relu1", new Relu());
        layers.put("Affine2", new Affine(params.get("w2"), params.get("b2")));

        lastLayer = new SoftmaxWithLoss();
    }

    private RealMatrix predict(RealMatrix x) {
        for (Layer layer : layers.values())
            x = layer.forward(x);

        return x;
    }

    public double loss(RealMatrix x, RealMatrix t) {
        RealMatrix y = predict(x);
        return lastLayer.forward(y, t);
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

    public HashMap<String, RealMatrix> gradient(RealMatrix x, RealMatrix t) {
        loss(x, t);

        RealMatrix dout = createMatrix(1, 1, 1.0);
        dout = lastLayer.backward(dout);

        List<String> keyList = new ArrayList<>(layers.keySet());
        Collections.reverse(keyList);
        for (String key : keyList)
            dout = layers.get(key).backward(dout);

        HashMap<String, RealMatrix> grads = new HashMap<>();
        grads.put("w1", layers.get("Affine1").dw());
        grads.put("b1", layers.get("Affine1").db());
        grads.put("w2", layers.get("Affine2").dw());
        grads.put("b2", layers.get("Affine2").db());

        return grads;
    }

    public void update(){
        layers.get("Affine1").update(params.get("w1"), params.get("b1"));
        layers.get("Affine2").update(params.get("w2"), params.get("b2"));
    }
}
