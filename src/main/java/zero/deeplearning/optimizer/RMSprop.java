package zero.deeplearning.optimizer;

import org.apache.commons.math3.linear.RealMatrix;

import java.util.HashMap;
import java.util.Map;

import static zero.deeplearning.common.Functions.*;
import static zero.deeplearning.common.Utils.createMatrix;

public class RMSprop implements Optimizer {
    private double lr;
    private double decayRate;
    private Map<String, RealMatrix> h;

    public RMSprop() {
        this(0.01, 0.99);
    }

    public RMSprop(double lr, double decayRate) {
        this.lr = lr;
        this.decayRate = decayRate;
    }

    public RMSprop(Map<String, Double> param) {
        this();
        if (param.containsKey("lr")) {
            lr = param.get("lr");
        }
        if (param.containsKey("decayRate")) {
            decayRate = param.get("decayRate");
        }
    }

    @Override
    public void update(Map<String, RealMatrix> params, Map<String, RealMatrix> grads) {
        if (h == null) {
            h = new HashMap<>();
            for (String key : params.keySet()) {
                h.put(key, createMatrix(params.get(key).getRowDimension(), params.get(key).getRowDimension()));
            }
        }

        for (String key : params.keySet()) {
            h.put(key, mult(h.get(key), decayRate));
            h.put(key, sub(h.get(key), mult(mult(grads.get(key), grads.get(key)), (1 - decayRate))));
            params.put(key, add(params.get(key), mult(h.get(key), decayRate * decayRate)));
            params.put(key, sub(params.get(key), mult(div(grads.get(key), add(sqrt(h.get(key)), 1e-7)), lr)));
        }
    }
}
