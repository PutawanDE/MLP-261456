package com.Tanadol.MLP;

import java.io.*;
import java.util.Scanner;

public class Main {
    private static final MathFunction leakyReluFn = (x) -> {
        if (x <= 0) return 0.01 * x;
        else return x;
    };

    private static final MathFunction diffLeakyReluFn = (x) -> {
        if (x <= 0) return 0.01;
        else return 1;
    };

    public static void main(String[] args) {
        trainFloodData();
    }

    public static void trainFloodData() {
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
            for (int i = 1; i <= 10; i++) {
                double[][] trainingData = readTrainingFloodDataset(foldSize, dataCols, i, k, path, delimiters);
                network.trainFloodData(trainingData, 0.9, 0.1, 400,
                        0.0002, "Flood_RMSE/FloodTrainingResult_D/ResultItr" + i);

                network.evaluateInput(trainingData, evalResultStr);
                network.evaluateInput(readTestFloodData(foldSize, dataCols, i, path, delimiters),
                        evalResultStr);
                evalResultStr.append('\n');
            }
            System.out.println(evalResultStr);
//            saveResult(evalResultStr, "Flood_RMSE/FloodTrainingResult_D/Result_D");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void saveResult(StringBuilder stringBuilder, String name) {
        File file = new File("D:/PUTAWAN/ComputerProjects/CI/" + name + ".csv");
        try (BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(file))) {
            bufferedWriter.append(stringBuilder);
        } catch (IOException exception) {
            exception.printStackTrace();
        }
    }

    private static double[][] readTrainingFloodDataset(int foldSize, int dataCols, int holdOutGroup, int k,
                                                       String path, String delimiters)
            throws FileNotFoundException {
        int totalFolds = k - 1;
        double[][] result = new double[foldSize * totalFolds][dataCols];

        int row = 0;
        for (int i = 1; i <= k; i++) {
            if (i == holdOutGroup) {
                continue;
            }

            Scanner scanner = new Scanner(new File(path + "/Fold" + i + ".txt"));
            scanner.useDelimiter(delimiters);
            int col = -1;

            while (scanner.hasNextInt()) {
                double num = scanner.nextInt();
                col++;
                if (col >= dataCols) {
                    col = 0;
                    row++;
                }
                result[row][col] = num;
            }
            scanner.close();
            row++;
        }

        return result;
    }

    private static double[][] readTestFloodData(int foldSize, int dataCols, int testGroup, String path, String delimiters) throws FileNotFoundException {
        Scanner scanner = new Scanner(new File(path + "/Fold" + testGroup + ".txt"));
        scanner.useDelimiter(delimiters);
        double[][] result = new double[foldSize][dataCols];
        int col = -1;
        int row = 0;

        while (scanner.hasNextInt()) {
            double num = scanner.nextInt();
            col++;
            if (col >= dataCols) {
                col = 0;
                row++;
            }
            result[row][col] = num;
        }

        return result;
    }
}
