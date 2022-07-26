package com.Tanadol.MLP;

import java.util.Random;

interface MathFunction {
    double run(double x);
}

public class Network {
    private int layerCount;
    private int[] nodeInLayerCount;
    private int nodeCount;

    private Matrix[] activations;
    private Matrix[] weights;
    private Matrix[] grads;
    private Matrix[] nets;
    private Matrix[] lastDeltaWeights;
    private Matrix outputLayerErrMat;

    private double[] desiredOutput = {
            1.0
    };
    private double sse;

    private final double MIN_WEIGTH = -1.0;
    private final double MAX_WEIGHT = 1.0;

    // f(x)=max(0, x)
    private final MathFunction reluFn = (x) -> {
        if (x < 0) return 0;
        else return x;
    };

    private final MathFunction diffReluFn = (x) -> {
        if (x < 0) return 0;
        else return 1;
    };

    private static final Random random = new Random();

    public Network(int[] nodeInLayerCount) {
        this.layerCount = nodeInLayerCount.length;
        this.nodeInLayerCount = new int[layerCount];
        for (int i = 0; i < nodeInLayerCount.length; i++) {
            this.nodeInLayerCount[i] = nodeInLayerCount[i];
            this.nodeCount += nodeInLayerCount[i];
        }

        weights = new Matrix[layerCount - 1];
        lastDeltaWeights = new Matrix[layerCount - 1];
        activations = new Matrix[layerCount];
        grads = new Matrix[layerCount];
        nets = new Matrix[layerCount];
    }

    public void train(double[][] dataset, double momentumRate, double learningRate) {
        min_max_norm(dataset);
        initWeight();

        for (double[] row : dataset) {
            feedForward(row);
            backProp(momentumRate, learningRate);
        }
    }

    // TODO: He Weight Init if training is bad
    private void initWeight() {
        for (int k = 0; k < weights.length; k++) {
            weights[k] = new Matrix(nodeInLayerCount[k + 1], nodeInLayerCount[k]);
            lastDeltaWeights[k] = new Matrix(nodeInLayerCount[k + 1], nodeInLayerCount[k]);
            for (int j = 0; j < weights[k].getRows(); j++) {
                for (int i = 0; i < weights[k].getCols(); i++) {
                    weights[k].data[j][i] = random.nextDouble(MIN_WEIGTH, MAX_WEIGHT);
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
            Matrix net = Matrix.multiply(weights[i - 1], (activations[i - 1]));
            activations[i] = Matrix.applyFunction(net, reluFn);
            nets[i] = net;
        }
    }

    private void backProp(double momentumRate, double learningRate) {
        calcErrors();
        calcGrads();

        // update weights
        for (int l = 0; l < layerCount - 1; l++) {
            for (int j = 0; j < nodeInLayerCount[l + 1]; j++) {
                for (int i = 0; i < nodeInLayerCount[l]; i++) {
                    double deltaWeight = momentumRate * lastDeltaWeights[l].data[j][i] +
                            learningRate * grads[l + 1].data[j][0] * activations[l].data[i][0];
                    weights[l].data[j][i] = weights[l].data[j][i] + deltaWeight;
                    lastDeltaWeights[l].data[j][i] = deltaWeight;
                }
            }
        }
    }

    private void calcGrads() {
        // output layer
        grads[layerCount - 1] = new Matrix(nodeInLayerCount[layerCount - 1], 1);
        for (int i = 0; i < nodeInLayerCount[layerCount - 1]; i++) {
            grads[layerCount - 1].data[i][0] = outputLayerErrMat.data[i][0] * diffReluFn.run(nets[layerCount - 1]
                    .data[i][0]);
        }

        // hidden layers
        for (int l = layerCount - 2; l >= 1; l--) {
            grads[l] = new Matrix(nodeInLayerCount[l], 1);
            for (int j = 0; j < nodeInLayerCount[l]; j++) {
                double sumGradsWeight = 0;
                for (int k = 0; k < nodeInLayerCount[l + 1]; k++) {
                    sumGradsWeight += grads[l + 1].data[k][0] * weights[l].data[k][j];
                }
                grads[l].data[j][0] = diffReluFn.run(nets[l].data[j][0]) * sumGradsWeight;
            }
        }
    }

    // calculate SSE and output layer error
    private void calcErrors() {
        outputLayerErrMat = new Matrix(desiredOutput.length, 1);
        for (int i = 0; i < desiredOutput.length; i++) {
            double error = desiredOutput[i] - activations[layerCount - 1].data[i][0];
            outputLayerErrMat.data[i][0] = error;
            sse = sse + (error * error);
        }
    }
}
