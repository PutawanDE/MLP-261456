package com.Tanadol.MLP;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class FloodDataNetwork extends Network {

    private final int inputLength;
    private final int desiredOutputLength;

    private StringBuilder stringBuilder;

    public FloodDataNetwork(int[] nodeInLayerCount, MathFunction activationFn, MathFunction diffActivationFn,
                            double minWeight, double maxWeight) {
        super(nodeInLayerCount, activationFn, diffActivationFn, minWeight, maxWeight);

        stringBuilder = new StringBuilder();
        inputLength = nodeInLayerCount[0];
        desiredOutputLength = nodeInLayerCount[nodeInLayerCount.length - 1];
    }

    public void trainFloodData(double[][] dataset, double momentumRate, double learningRate, int maxEpoch, double epsilon,
                               String name) {
        min_max_norm(dataset);

        double[][] inputData = copy2Darray(dataset, 0, inputLength);
        double[][] desiredOutputs = copy2Darray(dataset, inputLength, desiredOutputLength);

        trainData(inputData, desiredOutputs, momentumRate, learningRate, maxEpoch, epsilon, stringBuilder);
//        saveResult(name);
        stringBuilder = new StringBuilder();
    }

    public void evaluateInput(double[][] dataset, StringBuilder evalStringSb) {
        double[] min_max = min_max_norm(dataset);
        double min = min_max[0];
        double max = min_max[1];
        double range = max - min;

        double[][] inputData = copy2Darray(dataset, 0, inputLength);
        double[][] desiredOutputs = copy2Darray(dataset, inputLength, desiredOutputLength);

        double sse = 0;
        double sse_water = 0;
        for (int i = 0; i < dataset.length; i++) {
            sse += feedForward(inputData[i], desiredOutputs[i]);

            double desiredOutput = dataset[i][dataset[0].length - 1];
            double rawOutput = activations[layerCount - 1].data[0][0];
            double waterLevelErr = (desiredOutput * range + min) - (rawOutput * range + min);
            sse_water += waterLevelErr * waterLevelErr;
        }

        double mse = sse / dataset.length;
        double rmse = Math.sqrt(mse);
        double mse_water = sse_water / dataset.length;
        double rmse_water = Math.sqrt(mse_water);

        evalStringSb.append(rmse).append(',').append(rmse_water).append(',');
//        saveResult(name);
//        System.out.println(evalStringSb);
    }

    private void saveResult(String name) {
        File file = new File("D:/PUTAWAN/ComputerProjects/CI/" + name + ".csv");
        try (BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(file))) {
            bufferedWriter.append(stringBuilder);
        } catch (IOException exception) {
            exception.printStackTrace();
        }
    }

    // min-max normalization between 0, 1
    // xnormalized = (x - xminimum) / range of x
    // normalize only input
    private double[] min_max_norm(double[][] dataset) {
        double max = Integer.MIN_VALUE;
        double min = Integer.MAX_VALUE;

        for (int i = 0; i < dataset.length; i++) {
            for (int j = 0; j < dataset[i].length; j++) {
                if (dataset[i][j] > max) max = dataset[i][j];
                if (dataset[i][j] < min) min = dataset[i][j];
            }
        }

        double range = max - min;
        for (int i = 0; i < dataset.length; i++) {
            for (int j = 0; j < dataset[i].length; j++) {
                dataset[i][j] = (dataset[i][j] - min) / range;
            }
        }

        return new double[]{min, max};
    }
}
