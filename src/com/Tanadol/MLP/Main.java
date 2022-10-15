package com.Tanadol.MLP;

import static com.Tanadol.MLP.ReadWriteTrainingData.readTestData;
import static com.Tanadol.MLP.ReadWriteTrainingData.readTrainingDataset;
import static com.Tanadol.MLP.ReadWriteTrainingData.saveResult;

public class Main {
    private static final MathFunction linearFn = (x) -> x;
    private static final MathFunction diffLinearFn = (x) -> 1.0;

    private static final MathFunction leakyReluFn = (x) -> {
        if (x <= 0) return 0.01 * x;
        else return x;
    };
    private static final MathFunction diffLeakyReluFn = (x) -> {
        if (x <= 0) return 0.01;
        else return 1;
    };

    private static final MathFunction tanhFn = (x) -> 2.0 / (1 + Math.exp(-2.0 * x)) - 1.0;
    private static final MathFunction diffTanhFn = (x) -> 1.0 - tanhFn.run(x) * tanhFn.run(x);

    private static final MathFunction sigmoidFn = (x) -> 1.0 / (1.0 + Math.exp(x));
    private static final MathFunction diffSigmoidFn = (x) -> {
        double s = sigmoidFn.run(x);
        return s * (1.0 - s);
    };


    public static void main(String[] args) {
        tenPercentCrossValidateFloodData();
        tenPercentCrossValidateCrossPat();
    }

    public static void tenPercentCrossValidateFloodData() {
        int[] nodeInLayerCount = new int[]{8, 2, 1};
        MathFunction[] activationFn = new MathFunction[]{leakyReluFn, diffLeakyReluFn};
        Matrix[] biases = new Matrix[2];
        biases[0] = new Matrix(2, 1);
        biases[1] = new Matrix(1, 1);
        FloodDataNetwork network = new FloodDataNetwork(nodeInLayerCount, activationFn, activationFn,
                -1.0, 1.0, biases);

        int k = 10;
        int foldSize = 31;
        int dataCols = 9;
        String path = "D:/PUTAWAN/ComputerProjects/CI/HW1-mlp/Flood_dataset";
        String delimiters = "\\s*[\t\n]\\s*";

        try {
            StringBuilder evalResultStr = new StringBuilder();
            evalResultStr.append("Training RMSE,Training Water RMSE,TestRMSE,Test Water RMSE\n");
            for (int i = 1; i <= k; i++) {
                double[][] trainingData = readTrainingDataset(foldSize, dataCols, i, k, path, delimiters);
                network.trainFloodData(trainingData, 0.9, 0.1, 400,
                        0.0002, "Flood_RMSE/FloodTrainingResult_E/ResultItr" + i);

                network.evaluateInput(trainingData, evalResultStr);
                network.evaluateInput(readTestData(foldSize, dataCols, i, path, delimiters),
                        evalResultStr);
                evalResultStr.append('\n');
            }
            System.out.println(evalResultStr);
            saveResult(evalResultStr, "Flood_RMSE/FloodTrainingResult_D/Result_D");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void tenPercentCrossValidateCrossPat() {
        int[] nodeInLayerCount = new int[]{2, 2, 2, 1};
        Matrix[] biases = new Matrix[3];
        biases[0] = new Matrix(2, 1);
        biases[1] = new Matrix(2, 1);
        biases[2] = new Matrix(1, 1);

        MathFunction[] hiddenLayerActivation = new MathFunction[]{tanhFn, diffTanhFn};
        CrossPatDataNetwork network = new CrossPatDataNetwork(nodeInLayerCount, hiddenLayerActivation,
                -1.0, 1.0, biases);

        int k = 10;
        int foldSize = 20;
        int dataCols = 4;
        String path = "D:/PUTAWAN/ComputerProjects/CI/HW1-mlp/CrossPat_dataset";
        String delimiters = "\\s*[\\s\\n]\\s*";

        try {
            StringBuilder evalResultStr = new StringBuilder();
            int test_tp = 0, test_tn = 0, test_fp = 0, test_fn = 0;
            for (int i = 1; i <= k; i++) {
                double[][] trainingData = readTrainingDataset(foldSize, dataCols, i, k, path, delimiters);
                network.trainCrossPatData(trainingData, 0.01, 0.002, 5000,
                        0.01, "CrossPatResult/CrossPatTrainingResult_A/ResultItr" + i);

                evalResultStr.append("Training Data Confusion Matrix: ").append(i).append('\n');
                network.evaluateInput(trainingData, evalResultStr);
                evalResultStr.append("Test Data Confusion Matrix: ").append(i).append('\n');
                int[] testConfusionMat = network.evaluateInput(readTestData(foldSize, dataCols, i, path, delimiters),
                        evalResultStr);
                test_tp += testConfusionMat[0];
                test_fp += testConfusionMat[1];
                test_fn += testConfusionMat[2];
                test_tn += testConfusionMat[3];

                evalResultStr.append('\n');
            }
            System.out.println(evalResultStr);

            evalResultStr = new StringBuilder();
            evalResultStr.append(test_tn).append(',').append(test_fp).append("],[").append(test_fn).append(',').append(test_tp);
            System.out.println(evalResultStr);
            saveResult(evalResultStr, "CrossPatResult/Result_A");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
