package com.Tanadol.MLP;

import java.io.*;
import java.util.Scanner;

public class ReadWriteTrainingData {
    public static void saveResult(StringBuilder stringBuilder, String name) {
        File file = new File("D:/PUTAWAN/ComputerProjects/CI/" + name + ".csv");
        try (BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(file))) {
            bufferedWriter.append(stringBuilder);
        } catch (IOException exception) {
            exception.printStackTrace();
        }
    }

    public static double[][] readTrainingDataset(int foldSize, int dataCols, int holdOutGroup, int k,
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

            while (scanner.hasNextInt() || scanner.hasNextDouble()) {
                double num = 0;
                if (scanner.hasNextInt()) {
                    num = scanner.nextInt();
                } else if (scanner.hasNextDouble()) {
                    num = scanner.nextDouble();
                }

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

    public static double[][] readTestData(int foldSize, int dataCols, int testGroup, String path, String delimiters)
            throws FileNotFoundException {
        Scanner scanner = new Scanner(new File(path + "/Fold" + testGroup + ".txt"));
        scanner.useDelimiter(delimiters);
        double[][] result = new double[foldSize][dataCols];
        int col = -1;
        int row = 0;

        while (scanner.hasNextInt() || scanner.hasNextDouble()) {
            double num = 0;
            if (scanner.hasNextInt()) {
                num = scanner.nextInt();
            } else if (scanner.hasNextDouble()) {
                num = scanner.nextDouble();
            }

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
