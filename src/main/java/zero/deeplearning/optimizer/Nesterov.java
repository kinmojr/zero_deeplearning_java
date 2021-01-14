package zero.deeplearning.optimizer;

import org.apache.commons.math3.linear.RealMatrix;

import java.util.HashMap;
import java.util.Map;

import static zero.deeplearning.common.Functions.*;
import static zero.deeplearning.common.Utils.createMatrix;

public class Nesterov implements Optimizer {
    private double lr;
    private double momentum;
    private Map<String, RealMatrix> v;

    public Nesterov() {
        this(0.01, 0.9);
    }

    public Nesterov(double lr, double momentum) {
        this.lr = lr;
        this.momentum = momentum;
    }

    public Nesterov(Map<String, Double> param) {
        this();
        if (param.containsKey("lr")) {
            lr = param.get("lr");
        }
        if (param.containsKey("momentum")) {
            momentum = param.get("momentum");
        }
    }

    @Override
    public void update(Map<String, RealMatrix> params, Map<String, RealMatrix> grads) {
        if (v == null) {
            v = new HashMap<>();
            for (String key : params.keySet()) {
                v.put(key, createMatrix(params.get(key).getRowDimension(), params.get(key).getRowDimension()));
            }
        }

        for (String key : params.keySet()) {
            v.put(key, mult(v.get(key), momentum));
            v.put(key, sub(v.get(key), mult(grads.get(key), lr)));
            params.put(key, add(params.get(key), mult(v.get(key), momentum * momentum)));
            params.put(key, sub(params.get(key), mult(grads.get(key), (1 + momentum) * lr)));
        }
    }
}
