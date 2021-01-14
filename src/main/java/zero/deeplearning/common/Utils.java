package zero.deeplearning.common;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import zero.deeplearning.optimizer.*;

import java.io.*;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class Utils {
    public static int argmax(double[] values) {
        double maxValue = Double.NaN;
        int maxIndex = 0;
        for (int i = 0; i < values.length; i++) {
            if (Double.isNaN(maxValue)) {
                maxValue = values[i];
            } else if (values[i] > maxValue) {
                maxValue = values[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public static int[] argmax(double[][] values) {
        int[] maxIndices = new int[values.length];
        for (int i = 0; i < values.length; i++) {
            maxIndices[i] = argmax(values[i]);
        }
        return maxIndices;
    }

    public static double max(double[] values) {
        double max = Double.NaN;
        for (double value : values) {
            if (Double.isNaN(max)) {
                max = value;
            } else if (value > max) {
                max = value;
            }
        }
        return max;
    }

    public static RealMatrix readWeights(String file) throws IOException {
        List<String> lines = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(ClassLoader.getSystemResourceAsStream(file)))) {
            String line;
            while ((line = br.readLine()) != null) {
                lines.add(line);
            }
        }

        double[][] matrix = new double[lines.size()][lines.get(0).split("\t").length];
        for (int i = 0; i < matrix.length; i++) {
            String[] values = lines.get(i).split("\t");
            for (int j = 0; j < values.length; j++) {
                matrix[i][j] = Double.valueOf(values[j]);
            }
        }

        return createMatrix(matrix);
    }

    public static RealMatrix initWeight(int rowSize, int colSize, double weightInitStd) {
        Random rand = new Random();
        double[][] vals = new double[rowSize][colSize];
        for (int i = 0; i < rowSize; i++) {
            for (int j = 0; j < colSize; j++) {
                vals[i][j] = weightInitStd * rand.nextGaussian();
            }
        }
        return createMatrix(vals);
    }

    public static RealMatrix extractRowCol(RealMatrix aM, int startRow, int endRow, int startCol, int endCol) {
        return aM.getSubMatrix(startRow, endRow, startCol, endCol);
    }

    public static RealMatrix extractRowCol(RealMatrix aM, int startRow, int endRow, int[] cols) {
        return extractCol(extractRow(aM, startRow, endRow), cols);
    }

    public static RealMatrix extractRow(RealMatrix aM, int startRow, int endRow) {
        return extractRowCol(aM, startRow, endRow, 0, aM.getColumnDimension() - 1);
    }

    public static RealMatrix extractRow(RealMatrix aM, int[] rows) {
        int[] cols = new int[aM.getColumnDimension()];
        for (int i = 0; i < aM.getColumnDimension(); i++) cols[i] = i;
        return aM.getSubMatrix(rows, cols);
    }

    public static RealMatrix extractCol(RealMatrix aM, int startCol, int endCol) {
        return extractRowCol(aM, 0, aM.getRowDimension() - 1, startCol, endCol);
    }

    public static RealMatrix extractCol(RealMatrix aM, int[] cols) {
        int[] rows = new int[aM.getRowDimension()];
        for (int i = 0; i < aM.getRowDimension(); i++) rows[i] = i;
        return aM.getSubMatrix(rows, cols);
    }

    public static int[] randomChoice(int trainSize, int batchSize) {
        Random rand = new Random();
        int[] indexes = new int[batchSize];
        for (int i = 0; i < batchSize; i++) {
            indexes[i] = rand.nextInt(trainSize);
        }
        return indexes;
    }

    public static RealMatrix createMatrix(int row, int col, double val) {
        double[][] matrix = new double[row][col];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                matrix[i][j] = val;
            }
        }
        return createMatrix(matrix);
    }

    public static RealMatrix createMatrix(int row, int col) {
        return MatrixUtils.createRealMatrix(row, col);
    }

    public static RealMatrix createMatrix(double[][] vals) {
        return MatrixUtils.createRealMatrix(vals);
    }

    public static RealMatrix randomMatrix(int rowNum, int colNum) {
        RealMatrix ret = createMatrix(rowNum, colNum);
        Random rand = new Random();
        for (int i = 0; i < rowNum; i++) {
            for (int j = 0; j < colNum; j++) {
                ret.setEntry(i, j, rand.nextGaussian());
            }
        }
        return ret;
    }

    public static void download(String baseUrl, String baseDir, String fileName) throws IOException {
        String filePath = baseDir + fileName;
        if (new File(filePath).exists()) return;

        System.out.println("Downloading " + fileName + " ... ");

        URL url = new URL(baseUrl + fileName);
        URLConnection conn = url.openConnection();
        File file = new File(filePath);
        try (InputStream in = conn.getInputStream();
             FileOutputStream out = new FileOutputStream(file, false)) {
            byte[] data = new byte[1024];
            while (true) {
                int ret = in.read(data);
                if (ret == -1) {
                    break;
                }
                out.write(data, 0, ret);
            }
        }
    }

    public static void debugMatrix(String key, RealMatrix m) {
        System.out.print(key + " " + m.getRowDimension() + " * " + m.getColumnDimension() + " ");
        double total = 0.0;
        for (int i = 0; i < m.getRowDimension(); i++) {
            for (int j = 0; j < m.getColumnDimension(); j++) {
                total += Math.abs(m.getEntry(i, j));
            }
        }
        total = total / ((m.getRowDimension() + 1) * (m.getColumnDimension() + 1));
        System.out.println(total);
    }

    public static Optimizer createOptimizer(String optimizerName, Map<String, Double> optimizerParam) {
        Optimizer optimizer;
        if ("sgd".equals(optimizerName.toLowerCase())) {
            optimizer = new SGD(optimizerParam);
        } else if ("momentum".equals(optimizerName.toLowerCase())) {
            optimizer = new Momentum(optimizerParam);
        } else if ("nesterov".equals(optimizerName.toLowerCase())) {
            optimizer = new Nesterov(optimizerParam);
        } else if ("adgrad".equals(optimizerName.toLowerCase())) {
            optimizer = new AdaGrad(optimizerParam);
        } else if ("rmsprop".equals(optimizerName.toLowerCase())) {
            optimizer = new RMSprop(optimizerParam);
        } else if ("adgm".equals(optimizerName.toLowerCase())) {
            optimizer = new Adam(optimizerParam);
        } else {
            throw new RuntimeException("Unsupported optimizer");
        }
        return optimizer;
    }
}
