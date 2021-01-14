package zero.deeplearning.optimizer;

import org.apache.commons.math3.linear.RealMatrix;

import java.util.Map;

public interface Optimizer {
    public void update(Map<String, RealMatrix> params, Map<String, RealMatrix> grads);
}
