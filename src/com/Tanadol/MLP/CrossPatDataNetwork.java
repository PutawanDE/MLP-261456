package com.Tanadol.MLP;

import static com.Tanadol.MLP.ReadWriteTrainingData.saveResult;

public class CrossPatDataNetwork extends Network {
    public CrossPatDataNetwork(int[] nodeInLayerCount, MathFunction activationFn, MathFunction diffActivationFn,
                               double minWeight, double maxWeight) {
        super(nodeInLayerCount, activationFn, diffActivationFn, minWeight, maxWeight);
    }

    public void trainCrossPatData(double[][] dataset, double momentumRate, double learningRate, int maxEpoch, double epsilon,
                                  String name) {
        StringBuilder resultStrBuilder = new StringBuilder();

        double[][] inputData = copy2Darray(dataset, 0, inputLength);
        double[][] desiredOutputs = copy2Darray(dataset, inputLength, desiredOutputLength);

        trainData(inputData, desiredOutputs, momentumRate, learningRate, maxEpoch, epsilon, resultStrBuilder);
//        saveResult(resultStrBuilder, name);
    }

    public void evaluateInput(double[][] dataset, StringBuilder evalStringSb) {
        double[][] inputData = copy2Darray(dataset, 0, inputLength);
        double[][] desiredOutputs = copy2Darray(dataset, inputLength, desiredOutputLength);

        double tp = 0, tn = 0, fp = 0, fn = 0;
        for (int i = 0; i < dataset.length; i++) {
            feedForward(inputData[i], desiredOutputs[i]);

            double predicted1 = activations[layerCount - 1].data[0][0];
            double predicted2 = activations[layerCount - 1].data[1][0];

            // 1 for positive, 0 for negative, positive->first output is 1
            int predictedPositiveOrNegative = predicted1 > predicted2 ? 1 : 0;
            int actualPositiveOrNegative = (int) desiredOutputs[i][0];

            if (actualPositiveOrNegative == 1 && predictedPositiveOrNegative == 1) {
                tp++;
            } else if (actualPositiveOrNegative == 1 && predictedPositiveOrNegative == 0) {
                fn++;
            } else if (actualPositiveOrNegative == 0 && predictedPositiveOrNegative == 1) {
                fp++;
            } else if (actualPositiveOrNegative == 0 && predictedPositiveOrNegative == 0) {
                tn++;
            }
        }

        evalStringSb.append(",actually positive (1),").append("actually negative (0)\n");
        evalStringSb.append("predicted positive (1)").append(tp).append(',').append(fp).append('\n');
        evalStringSb.append("predicted negative (0)").append(fn).append(',').append(tn).append('\n');
        System.out.println(evalStringSb);
    }
}