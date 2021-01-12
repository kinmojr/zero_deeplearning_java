package zero.deeplearning.common.optimizer;

import org.apache.commons.math3.linear.RealMatrix;

import java.util.LinkedHashMap;
import java.util.Map;

import static zero.deeplearning.common.Functions.*;
import static zero.deeplearning.common.Utils.*;

public class AdaGrad implements Optimizer {
    private double lr;
    private Map<String, RealMatrix> h;

    public AdaGrad() {
        this(0.01);
    }

    public AdaGrad(double lr) {
        this.lr = lr;
    }

    @Override
    public void update(Map<String, RealMatrix> params, Map<String, RealMatrix> grads) {
        if (h == null) {
            h = new LinkedHashMap<>();
            for (String key : params.keySet()) {
                h.put(key, createMatrix(params.get(key).getRowDimension(), params.get(key).getColumnDimension()));
            }
        }

        for (String key : params.keySet()) {
            h.put(key, add(h.get(key), mult(grads.get(key), grads.get(key))));
            params.put(key, sub(params.get(key), mult(div(grads.get(key), add(sqrt(h.get(key)), 1e-7)), lr)));
        }
    }
}
