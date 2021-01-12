package zero.deeplearning.common.optimizer;

import org.apache.commons.math3.linear.RealMatrix;

import java.util.LinkedHashMap;
import java.util.Map;

import static zero.deeplearning.common.Functions.*;
import static zero.deeplearning.common.Utils.createMatrix;

public class Momentum implements Optimizer {
    private double lr;
    private double momentum;
    private Map<String, RealMatrix> v;

    public Momentum() {
        this(0.01);
    }

    public Momentum(double lr) {
        this(lr, 0.9);
    }

    public Momentum(double lr, double momentum) {
        this.lr = lr;
        this.momentum = momentum;
    }

    @Override
    public void update(Map<String, RealMatrix> params, Map<String, RealMatrix> grads) {
        if (v == null) {
            v = new LinkedHashMap<>();
            for (String key : params.keySet()) {
                v.put(key, createMatrix(params.get(key).getRowDimension(), params.get(key).getColumnDimension()));
            }
        }

        for (String key : params.keySet()) {
            v.put(key, sub(mult(v.get(key), momentum), mult(grads.get(key), lr)));
            params.put(key, add(params.get(key), v.get(key)));
        }
    }
}
