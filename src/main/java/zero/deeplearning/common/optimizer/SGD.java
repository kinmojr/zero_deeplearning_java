package zero.deeplearning.common.optimizer;

import org.apache.commons.math3.linear.RealMatrix;

import java.util.Map;

import static zero.deeplearning.common.Functions.*;

public class SGD implements Optimizer {
    private double lr;

    public SGD() {
        this(0.01);
    }

    public SGD(double lr) {
        this.lr = lr;
    }

    @Override
    public void update(Map<String, RealMatrix> params, Map<String, RealMatrix> grads) {
        for (String key : params.keySet()) {
            params.put(key, sub(params.get(key), mult(grads.get(key), lr)));
        }
    }
}
