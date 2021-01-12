package zero.deeplearning.common;

import static zero.deeplearning.common.Utils.*;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class Functions {
    public static RealMatrix add(RealMatrix aM, RealMatrix bM) {
        return aM.add(bM);
    }

    public static RealMatrix addBias(RealMatrix aM, RealMatrix bM) {
        double[] row = bM.getRow(0);
        double[][] b = new double[aM.getRowDimension()][row.length];
        for (int i = 0; i < aM.getRowDimension(); i++) {
            b[i] = row;
        }
        return add(aM, createMatrix(b));
    }

    public static RealMatrix add(RealMatrix aM, double b) {
        return aM.scalarAdd(b);
    }

    public static RealVector add(RealVector aV, double b) {
        return aV.mapAdd(b);
    }

    public static RealMatrix sub(RealMatrix aM, RealMatrix bM) {
        return aM.subtract(bM);
    }

    public static RealVector sub(RealVector aV, RealVector bV) {
        return aV.subtract(bV);
    }

    public static RealMatrix mult(RealMatrix aM, double b) {
        return aM.scalarMultiply(b);
    }

    public static RealMatrix mult(RealMatrix aM, RealMatrix bM) {
        double[][] aA2 = aM.getData();
        double[][] bA2 = bM.getData();
        double[][] retA2 = new double[aA2.length][aA2[0].length];
        for (int i = 0; i < aA2.length; i++) {
            for (int j = 0; j < aA2[i].length; j++) {
                retA2[i][j] = aA2[i][j] * bA2[i][j];
            }
        }
        return createMatrix(retA2);
    }

    public static RealVector mult(RealVector aV, double b) {
        return aV.mapMultiply(b);
    }

    public static RealMatrix div(RealMatrix aM, double b) {
        return aM.scalarMultiply(1 / b);
    }

    public static RealMatrix div(RealMatrix aM, RealMatrix bM) {
        double[][] aA2 = aM.getData();
        double[][] bA2 = bM.getData();
        double[][] retA2 = new double[aA2.length][aA2[0].length];
        for (int i = 0; i < aA2.length; i++) {
            for (int j = 0; j < aA2[i].length; j++) {
                retA2[i][j] = aA2[i][j] / bA2[i][j];
            }
        }
        return createMatrix(retA2);
    }

    public static RealMatrix dot(RealMatrix aM, RealMatrix bM) {
        return aM.multiply(bM);
    }

    public static RealVector dot(RealMatrix aM, RealVector bV) {
        return aM.operate(bV);
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

    public static double mean(RealVector aV) {
        double[] aA = aV.toArray();
        double sum = 0.0;
        for (double val : aA) {
            sum += val;
        }
        return sum / aA.length;
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

}
