package com.Tanadol.MLP;

import static com.Tanadol.MLP.ReadWriteTrainingData.saveResult;

public class FloodDataNetwork extends Network {
    public FloodDataNetwork(int[] nodeInLayerCount, MathFunction[] hiddenLayerActivation,
                            MathFunction[] outputLayerActivation, double minWeight, double maxWeight) {
        super(NETWORK_TYPE.REGRESSION, nodeInLayerCount, hiddenLayerActivation, outputLayerActivation, minWeight, maxWeight);
    }

    public void trainFloodData(double[][] dataset, double momentumRate, double learningRate, int maxEpoch, double epsilon,
                               String name) {
        StringBuilder resultStrBuilder = new StringBuilder();
        min_max_norm(dataset);

        double[][] inputData = copy2Darray(dataset, 0, inputLength);
        double[][] desiredOutputs = copy2Darray(dataset, inputLength, desiredOutputLength);

        trainData(inputData, desiredOutputs, momentumRate, learningRate, maxEpoch, epsilon, resultStrBuilder);
        saveResult(resultStrBuilder, name);
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
//        saveResult(evalStringSb, name);
//        System.out.println(evalStringSb);
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
