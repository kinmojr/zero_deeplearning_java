package zero.deeplearning.optimizer;

import org.apache.commons.math3.linear.RealMatrix;

import java.util.HashMap;
import java.util.Map;

import static zero.deeplearning.common.Functions.*;
import static zero.deeplearning.common.Utils.createMatrix;

public class Adam implements Optimizer {
    private double lr;
    private double beta1;
    private double beta2;
    private int iter = 0;
    private Map<String, RealMatrix> m;
    private Map<String, RealMatrix> v;

    public Adam() {
        this(0.01);
    }

    public Adam(double lr) {
        this(lr, 0.9, 0.999);
    }

    public Adam(double lr, double beta1, double beta2) {
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
    }

    public Adam(Map<String, Double> param) {
        this();
        if (param.containsKey("lr")) {
            lr = param.get("lr");
        }
        if (param.containsKey("beta1")) {
            beta1 = param.get("beta1");
        }
        if (param.containsKey("beta2")) {
            beta2 = param.get("beta2");
        }
    }

    @Override
    public void update(Map<String, RealMatrix> params, Map<String, RealMatrix> grads) {
        if (m == null) {
            m = new HashMap<String, RealMatrix>();
            v = new HashMap<String, RealMatrix>();
            for (String key : params.keySet()) {
                m.put(key, createMatrix(params.get(key).getRowDimension(), params.get(key).getColumnDimension()));
                v.put(key, createMatrix(params.get(key).getRowDimension(), params.get(key).getColumnDimension()));
            }
        }

        iter++;
        double lrt = lr * Math.sqrt(1.0 - Math.pow(beta2, iter)) / (1.0 - Math.pow(beta1, iter));

        for (String key : params.keySet()) {
            m.put(key, add(m.get(key), mult(sub(grads.get(key), m.get(key)), 1 - beta1)));
            v.put(key, add(v.get(key), mult(sub(pow(grads.get(key), 2.0), v.get(key)), 1 - beta2)));

            params.put(key, sub(params.get(key), div(mult(m.get(key), lrt), add(sqrt(v.get(key)), 1e-7))));
        }
    }
}
