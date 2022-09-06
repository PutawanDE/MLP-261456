package com.Tanadol.MLP;

import static com.Tanadol.MLP.ReadWriteTrainingData.readTestData;
import static com.Tanadol.MLP.ReadWriteTrainingData.readTrainingDataset;
import static com.Tanadol.MLP.ReadWriteTrainingData.saveResult;

public class Main {
    private static final MathFunction leakyReluFn = (x) -> {
        if (x <= 0) return 0.01 * x;
        else return x;
    };

    private static final MathFunction diffLeakyReluFn = (x) -> {
        if (x <= 0) return 0.01;
        else return 1;
    };

    private static final MathFunction sigmoidFn = (x) -> 1.0 / (1.0 + Math.exp(x));

    private static final MathFunction diffSigmoidFn = (x) -> {
        double s = sigmoidFn.run(x);
        return s * (1.0 - s);
    };


    public static void main(String[] args) {
//        tenPercentCrossValidateFloodData();
        tenPercentCrossValidateCrossPat();
    }

    public static void tenPercentCrossValidateFloodData() {
        int[] nodeInLayerCount = new int[]{8, 5, 1};
        FloodDataNetwork network = new FloodDataNetwork(nodeInLayerCount, leakyReluFn, diffLeakyReluFn,
                -1.0, 1.0);

        int k = 10;
        int foldSize = 31;
        int dataCols = 9;
        String path = "D:/PUTAWAN/ComputerProjects/CI/Flood_dataset";
        String delimiters = "\\s*[\t\n]\\s*";

        try {
            StringBuilder evalResultStr = new StringBuilder();
            evalResultStr.append("Training RMSE,Training Water RMSE,TestRMSE,Test Water RMSE\n");
            for (int i = 1; i <= k; i++) {
                double[][] trainingData = readTrainingDataset(foldSize, dataCols, i, k, path, delimiters);
                network.trainFloodData(trainingData, 0.9, 0.1, 400,
                        0.0002, "Flood_RMSE/FloodTrainingResult_D/ResultItr" + i);

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
        int[] nodeInLayerCount = new int[]{2, 2, 2};
        CrossPatDataNetwork network = new CrossPatDataNetwork(nodeInLayerCount, sigmoidFn, diffSigmoidFn,
                0.0, 1.0);

        int k = 10;
        int foldSize = 20;
        int dataCols = 4;
        String path = "D:/PUTAWAN/ComputerProjects/CI/CrossPat_dataset";
        String delimiters = "\\s*[\\s\\n]\\s*";

        try {
            StringBuilder evalResultStr = new StringBuilder();
            for (int i = 1; i <= k; i++) {
                double[][] trainingData = readTrainingDataset(foldSize, dataCols, i, k, path, delimiters);
                network.trainCrossPatData(trainingData, 0, 0.1, 4000,
                        0.0002, "CrossPatResult/CrossPatTrainingResult_A/ResultItr" + i);

//                evalResultStr.append("Training Data Confusion Matrix\n");
//                network.evaluateInput(trainingData, evalResultStr);
//                evalResultStr.append("Test Data Confusion Matrix\n");
//                network.evaluateInput(readTestData(foldSize, dataCols, i, path, delimiters),
//                        evalResultStr);
//                evalResultStr.append('\n');
            }
            System.out.println(evalResultStr);
//            saveResult(evalResultStr, "CrossPatResult/CrossPatResult_A/Result_A");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
