package zero.deeplearning.common;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.io.*;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.List;
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

        return MatrixUtils.createRealMatrix(matrix);
    }

    public static RealMatrix initWeight(int rowSize, int colSize, double weightInitStd) {
        Random rand = new Random();
        double[][] vals = new double[rowSize][colSize];
        for (int i = 0; i < rowSize; i++) {
            for (int j = 0; j < colSize; j++) {
                vals[i][j] = weightInitStd * rand.nextGaussian();
            }
        }
        return MatrixUtils.createRealMatrix(vals);
    }

    public static RealMatrix initBias(int size) {
        return MatrixUtils.createRealMatrix(new double[1][size]);
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

    public static RealVector extractRowCol(RealMatrix aM, int startRow, int endRow, int col) {
        return aM.getColumnVector(col).getSubVector(startRow, endRow - startRow + 1);
    }

    public static RealVector extract(RealVector aV, int start, int end) {
        return aV.getSubVector(start, end - start + 1);
    }

    public static int[] randomChoice(int trainSize, int batchSize) {
        Random rand = new Random();
        int[] indexes = new int[batchSize];
        for (int i = 0; i < batchSize; i++) {
            indexes[i] = rand.nextInt(trainSize);
        }
        return indexes;
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
}
