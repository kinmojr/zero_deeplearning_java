package zero.deeplearning.common;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.zip.GZIPInputStream;

import static zero.deeplearning.common.Utils.*;

public class MNIST {
    private static final String URL_BASE = "http://yann.lecun.com/exdb/mnist/";
    private static final HashMap<String, String> KEY_FILE = new HashMap<>();
    private static final String DATASET_DIR = "./dataset/mnist/";
    private static HashMap<String, RealMatrix> dataset;

    static {
        KEY_FILE.put("train_img", "train-images-idx3-ubyte.gz");
        KEY_FILE.put("train_label", "train-labels-idx1-ubyte.gz");
        KEY_FILE.put("test_img", "t10k-images-idx3-ubyte.gz");
        KEY_FILE.put("test_label", "t10k-labels-idx1-ubyte.gz");
    }

    public static HashMap<String, RealMatrix> loadMinist(boolean normalize, boolean oneHotEncode) throws
            IOException {
        initMnist();

        if (normalize) {
            normalize(dataset.get("train_img"));
            normalize(dataset.get("test_img"));
        }

        if (oneHotEncode) {
            dataset.put("train_label", oneHotEncode(dataset.get("train_label")));
            dataset.put("test_label", oneHotEncode(dataset.get("test_label")));
        }

        return dataset;
    }

    private static void initMnist() throws IOException {
        downloadMnist();
        dataset = new HashMap<>();
        dataset.put("train_img", loadImg(KEY_FILE.get("train_img")));
        dataset.put("train_label", loadLabel(KEY_FILE.get("train_label")));
        dataset.put("test_img", loadImg(KEY_FILE.get("test_img")));
        dataset.put("test_label", loadLabel(KEY_FILE.get("test_label")));
    }


    private static void downloadMnist() throws IOException {
        if (!new File(DATASET_DIR).exists()) new File(DATASET_DIR).mkdirs();
        for (String v : KEY_FILE.values()) {
            download(URL_BASE, DATASET_DIR, v);
        }
    }

    private static RealMatrix loadLabel(String fileName) throws IOException {
        double[][] data;
        String filePath = DATASET_DIR + "/" + fileName;
        try (DataInputStream is = new DataInputStream(new GZIPInputStream(new FileInputStream(filePath)))) {
            is.readInt();
            int numRow = is.readInt();
            data = new double[numRow][1];
            for (int i = 0; i < numRow; i++) {
                data[i][0] = is.readUnsignedByte();
            }
        }
        return MatrixUtils.createRealMatrix(data);
    }

    private static RealMatrix loadImg(String fileName) throws IOException {
        double[][] data;
        String filePath = DATASET_DIR + "/" + fileName;
        try (DataInputStream is = new DataInputStream(new GZIPInputStream(new FileInputStream(filePath)))) {
            is.readInt();
            int numRow = is.readInt();
            int numCol = is.readInt() * is.readInt();
            data = new double[numRow][numCol];
            for (int i = 0; i < numRow; i++) {
                for (int j = 0; j < numCol; j++) {
                    data[i][j] = is.readUnsignedByte();
                }
            }
        }
        return MatrixUtils.createRealMatrix(data);
    }

    private static void normalize(RealMatrix x) {
        for (int i = 0; i < x.getRowDimension(); i++) {
            for (int j = 0; j < x.getColumnDimension(); j++) {
                x.setEntry(i, j, x.getEntry(i, j) / 255.0);
            }
        }
    }

    private static RealMatrix oneHotEncode(RealMatrix x) {
        double[][] ret = new double[x.getRowDimension()][10];
        for (int i = 0; i < x.getRowDimension(); i++) {
            ret[i][(int) x.getEntry(i, 0)] = 1.0;
        }
        return MatrixUtils.createRealMatrix(ret);
    }
}
