package com.Tanadol.MLP;

import java.util.Random;

interface ActivationFunction {
    double run(double x);
}

public class Network {
    private int layerCount;
    private int[] nodeInLayerCount;
    private int nodeCount;

    private Matrix[] activations;

    private Matrix[] networkWeightsMat;

    private final double MIN_WEIGTH = -1.0;
    private final double MAX_WEIGHT = 1.0;

    // f(x)=max(0, x)
    private final ActivationFunction reluFn = (x) -> {
        if (x < 0) return 0;
        else return x;
    };

    private static final Random random = new Random();

    public Network(int[] nodeInLayerCount) {
        this.layerCount = nodeInLayerCount.length;
        this.nodeInLayerCount = new int[layerCount];
        for (int i = 0; i < nodeInLayerCount.length; i++) {
            this.nodeInLayerCount[i] = nodeInLayerCount[i];
            this.nodeCount += nodeInLayerCount[i];
        }

        networkWeightsMat = new Matrix[layerCount - 1];
        activations = new Matrix[layerCount];
    }

    public void train(double[][] dataset) {
        min_max_norm(dataset);

        for (double[] row : dataset) {
            initWeight();
            feedForward(row);
        }
    }

    // TODO: He Weight Init if training is bad
    private void initWeight() {
        for (int i = 0; i < networkWeightsMat.length; i++) {
            networkWeightsMat[i] = new Matrix(nodeInLayerCount[i + 1], nodeInLayerCount[i]);
            for (int j = 0; j < networkWeightsMat[i].getRows(); j++) {
                for (int k = 0; k < networkWeightsMat[i].getCols(); k++) {
                    networkWeightsMat[i].data[j][k] = random.nextDouble(MIN_WEIGTH, MAX_WEIGHT);
                }
            }
        }
    }

    // min-max normalization between 0, 1
    // xnormalized = (x - xminimum) / range of x
    private void min_max_norm(double[][] dataset) {
        double[] min = new double[dataset[0].length];
        double[] max = new double[dataset[0].length];

        for (int i = 0; i < dataset[0].length; i++) {
            min[i] = Integer.MAX_VALUE;
            max[i] = Integer.MIN_VALUE;

            for (int j = 0; j < dataset.length; j++) {
                double x = dataset[j][i];
                if (x < min[i]) min[i] = x;
                if (x > max[i]) max[i] = x;
            }
        }

        for (int j = 0; j < dataset.length; j++) {
            for (int i = 0; i < dataset[j].length; i++) {
                double range = max[i] - min[i];
                if (range != 0) dataset[j][i] = (dataset[j][i] - min[i]) / range;
                else dataset[j][i] = 0.0;
            }
        }
    }

    private void feedForward(double[] inputs) {
        double[][] inputVect = new double[inputs.length][1];
        for (int i = 0; i < inputs.length; i++) {
            inputVect[i][0] = inputs[i];
        }
        activations[0] = new Matrix(inputVect);
        for (int i = 1; i < layerCount; i++) {
            Matrix net = Matrix.multiply(networkWeightsMat[i - 1], (activations[i - 1]));
            activations[i] = Matrix.applyFunction(net, reluFn);
        }
    }
}
