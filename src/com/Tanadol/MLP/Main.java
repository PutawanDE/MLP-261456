package com.Tanadol.MLP;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        trainFloodData();
    }

    public static void trainFloodData() {
        int[] nodeInLayerCount = new int[]{8, 5, 1};
        Network network = new Network(nodeInLayerCount);

        int k = 10;
        int foldSize = 31;
        int dataCols = 9;
        String path = "D:/PUTAWAN/ComputerProjects/CI/Flood_dataset";
        String delimiters = "\\s*[\t\n]\\s*";

        try {
            for (int i = 1; i <= 10; i++) {
                double[][] trainingData = readTrainingFloodDataset(foldSize, dataCols, i, k, path, delimiters);
                network.train(trainingData, 0.004, 0.001, 400, "ResultItr" + i);
                network.evaluateInput(readTestFloodData(foldSize, dataCols, i, path, delimiters),
                        "ResultTestSet" + i);
            }
        } catch (Exception e) {
            e.printStackTrace();
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
