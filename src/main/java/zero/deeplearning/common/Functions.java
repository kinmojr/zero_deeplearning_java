package zero.deeplearning.common;

import org.apache.commons.math3.linear.RealMatrix;

import static zero.deeplearning.common.Utils.createMatrix;
import static zero.deeplearning.common.Utils.max;

public class Functions {
    public static RealMatrix add(RealMatrix aM, RealMatrix bM) {
        if (isSameSize(aM, bM)) {
            return aM.add(bM);
        } else if (canExpandFirst(aM, bM)) {
            return add(copyRow(aM, bM.getRowDimension()), bM);
        } else if (canExpandSecond(aM, bM)) {
            return add(aM, copyRow(bM, aM.getRowDimension()));
        } else {
            throw new RuntimeException("Invalid matrix size");
        }
    }

    public static RealMatrix add(RealMatrix aM, double b) {
        return aM.scalarAdd(b);
    }

    public static RealMatrix sub(RealMatrix aM, RealMatrix bM) {
        if (isSameSize(aM, bM)) {
            return aM.subtract(bM);
        } else if (canExpandFirst(aM, bM)) {
            return sub(copyRow(aM, bM.getRowDimension()), bM);
        } else if (canExpandSecond(aM, bM)) {
            return sub(aM, copyRow(bM, aM.getRowDimension()));
        } else {
            throw new RuntimeException("Invalid matrix size");
        }
    }

    public static RealMatrix mult(RealMatrix aM, double b) {
        return aM.scalarMultiply(b);
    }

    public static RealMatrix mult(RealMatrix aM, RealMatrix bM) {
        if (isSameSize(aM, bM)) {
            double[][] aA2 = aM.getData();
            double[][] bA2 = bM.getData();
            double[][] retA2 = new double[aA2.length][aA2[0].length];
            for (int i = 0; i < aA2.length; i++) {
                for (int j = 0; j < aA2[i].length; j++) {
                    retA2[i][j] = aA2[i][j] * bA2[i][j];
                }
            }
            return createMatrix(retA2);
        } else if (canExpandFirst(aM, bM)) {
            return mult(copyRow(aM, bM.getRowDimension()), bM);
        } else if (canExpandSecond(aM, bM)) {
            return mult(aM, copyRow(bM, aM.getRowDimension()));
        } else {
            throw new RuntimeException("Invalid matrix size");
        }
    }

    public static RealMatrix div(RealMatrix aM, double b) {
        return aM.scalarMultiply(1 / b);
    }

    public static RealMatrix div(RealMatrix aM, RealMatrix bM) {
        if (isSameSize(aM, bM)) {
            double[][] aA2 = aM.getData();
            double[][] bA2 = bM.getData();
            double[][] retA2 = new double[aA2.length][aA2[0].length];
            for (int i = 0; i < aA2.length; i++) {
                for (int j = 0; j < aA2[i].length; j++) {
                    retA2[i][j] = aA2[i][j] / bA2[i][j];
                }
            }
            return createMatrix(retA2);
        } else if (canExpandFirst(aM, bM)) {
            return div(copyRow(aM, bM.getRowDimension()), bM);
        } else if (canExpandSecond(aM, bM)) {
            return div(aM, copyRow(bM, aM.getRowDimension()));
        } else {
            throw new RuntimeException("Invalid matrix size");
        }
    }

    public static RealMatrix dot(RealMatrix aM, RealMatrix bM) {
        return aM.multiply(bM);
    }

    public static RealMatrix sumCol(RealMatrix aM) {
        double[][] vals = aM.getData();
        double[][] ret = new double[1][aM.getColumnDimension()];
        for (int j = 0; j < vals[0].length; j++) {
            double sum = 0.0;
            for (int i = 0; i < vals.length; i++) {
                sum += vals[i][j];
            }
            ret[0][j] = sum;
        }
        return createMatrix(ret);
    }

    public static RealMatrix pow(RealMatrix aM, double b) {
        double[][] pA2 = aM.getData();
        for (int i = 0; i < pA2.length; i++) {
            for (int j = 0; j < pA2[i].length; j++) {
                pA2[i][j] = Math.pow(pA2[i][j], b);
            }
        }
        return createMatrix(pA2);
    }

    public static RealMatrix sqrt(RealMatrix aM) {
        double[][] pA2 = aM.getData();
        for (int i = 0; i < pA2.length; i++) {
            for (int j = 0; j < pA2[i].length; j++) {
                pA2[i][j] = Math.sqrt(pA2[i][j]);
            }
        }
        return createMatrix(pA2);
    }

    public static RealMatrix t(RealMatrix aM) {
        return aM.transpose();
    }

    public static double crossEntropyError(double[] y, double[] t) {
        double sum = 0.0;
        for (int i = 0; i < y.length; i++) {
            sum -= t[i] * Math.log(y[i] + Math.pow(10.0, -7.0));
        }
        return sum;
    }

    public static double crossEntropyError(double[][] y, double[][] t) {
        double sum = 0.0;
        for (int i = 0; i < y.length; i++) {
            sum += crossEntropyError(y[i], t[i]);
        }
        return sum / y.length;
    }

    public static double crossEntropyError(RealMatrix y, RealMatrix t) {
        return crossEntropyError(y.getData(), t.getData());
    }

    public static double[] softmax(double[] values) {
        double max = max(values);

        double sum = 0.0;
        for (double value : values) {
            sum += Math.exp(value - max);
        }

        double[] ret = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            ret[i] = Math.exp(values[i] - max) / sum;
        }
        return ret;
    }

    public static double[][] softmax(double[][] values) {
        double[][] ret = new double[values.length][];
        for (int i = 0; i < values.length; i++) {
            ret[i] = softmax(values[i]);
        }
        return ret;
    }

    public static RealMatrix softmax(RealMatrix matrix) {
        return createMatrix(softmax(matrix.getData()));
    }

    public static double sigmoid(double value) {
        return 1.0 / (1.0 + Math.exp(-value));
    }

    public static double[] sigmoid(double[] values) {
        double[] ret = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            ret[i] = sigmoid(values[i]);
        }
        return ret;
    }

    public static double[][] sigmoid(double[][] values) {
        double[][] ret = new double[values.length][];
        for (int i = 0; i < values.length; i++) {
            ret[i] = sigmoid(values[i]);
        }
        return ret;
    }

    public static RealMatrix sigmoid(RealMatrix matrix) {
        return createMatrix(sigmoid(matrix.getData()));
    }

    public static RealMatrix sigmoidGrad(RealMatrix matrix) {
        RealMatrix ones = createMatrix(matrix.getRowDimension(), matrix.getColumnDimension(), 1.0);
        return mult(sub(ones, sigmoid(matrix)), sigmoid(matrix));
    }

    public static double l2norm(RealMatrix x) {
        double ret = 0.0;
        for (int i = 0; i < x.getRowDimension(); i++) {
            for (int j = 0; j < x.getColumnDimension(); j++) {
                ret += Math.pow(x.getEntry(i, j), 2);
            }
        }
        return ret;
    }

    private static boolean isSameSize(RealMatrix aM, RealMatrix bM) {
        if (aM.getRowDimension() == bM.getRowDimension() && aM.getColumnDimension() == bM.getColumnDimension()) {
            return true;
        }
        return false;
    }

    private static boolean canExpandFirst(RealMatrix aM, RealMatrix bM) {
        if (aM.getColumnDimension() == bM.getColumnDimension() && aM.getRowDimension() == 1) {
            return true;
        }
        return false;
    }

    private static boolean canExpandSecond(RealMatrix aM, RealMatrix bM) {
        if (aM.getColumnDimension() == bM.getColumnDimension() && bM.getRowDimension() == 1) {
            return true;
        }
        return false;
    }

    private static RealMatrix copyRow(RealMatrix aM, int num) {
        double[][] ret = new double[num][aM.getColumnDimension()];
        for (int i = 0; i < ret.length; i++) {
            for (int j = 0; j < ret[i].length; j++) {
                ret[i] = aM.getData()[0];
            }
        }
        return createMatrix(ret);
    }
}
