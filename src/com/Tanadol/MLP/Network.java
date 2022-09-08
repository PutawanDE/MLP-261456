package com.Tanadol.MLP;

import java.util.Random;

interface MathFunction {
    double run(double x);
}

enum NETWORK_TYPE {
    REGRESSION, BIN_CLASSIFIER
}

public class Network {
    protected final int inputLength;
    protected final int desiredOutputLength;
    private final NETWORK_TYPE type;

    protected int layerCount;
    private int[] nodeInLayerCount;

    protected Matrix[] activations;
    private Matrix[] weights;
    private Matrix[] grads;
    private Matrix[] nets;
    private Matrix[] lastDeltaWeights;
    private double[] diffLossVect;

    private final double minWeight;
    private final double maxWeight;

    private final MathFunction hiddenLayerActivationFn;
    private final MathFunction diffHiddenLayerActivationFn;
    private final MathFunction outputLayerActivationFn;
    private final MathFunction diffOutputLayerActivationFn;

    private static final Random random = new Random();

    public Network(NETWORK_TYPE type, int[] nodeInLayerCount, MathFunction[] hiddenLayerActivation,
                   MathFunction[] outputLayerActivation, double minWeight, double maxWeight) {
        this.layerCount = nodeInLayerCount.length;
        this.nodeInLayerCount = nodeInLayerCount;
        this.type = type;

        this.hiddenLayerActivationFn = hiddenLayerActivation[0];
        this.diffHiddenLayerActivationFn = hiddenLayerActivation[1];
        this.outputLayerActivationFn = outputLayerActivation[0];
        this.diffOutputLayerActivationFn = outputLayerActivation[1];

        this.minWeight = minWeight;
        this.maxWeight = maxWeight;

        weights = new Matrix[layerCount - 1];
        lastDeltaWeights = new Matrix[layerCount - 1];
        activations = new Matrix[layerCount];
        grads = new Matrix[layerCount];
        nets = new Matrix[layerCount];

        initWeight();

        inputLength = nodeInLayerCount[0];
        desiredOutputLength = nodeInLayerCount[nodeInLayerCount.length - 1];
        diffLossVect = new double[desiredOutputLength];
    }

    private void initWeight() {
        for (int k = 0; k < weights.length; k++) {
            weights[k] = new Matrix(nodeInLayerCount[k + 1], nodeInLayerCount[k]);
            lastDeltaWeights[k] = new Matrix(nodeInLayerCount[k + 1], nodeInLayerCount[k]);
            for (int j = 0; j < weights[k].getRows(); j++) {
                for (int i = 0; i < weights[k].getCols(); i++) {
                    weights[k].data[j][i] = random.nextDouble(minWeight, maxWeight);
                }
            }
        }
    }

    protected void trainData(double[][] inputs, double[][] desiredOutputs,
                             double momentumRate, double learningRate, int maxEpoch, double epsilon,
                             StringBuilder epochResult) {
        int e = 0;
        double loss;
        double epochLoss = Integer.MAX_VALUE;
        int n = inputs.length;

        while (e < maxEpoch && epochLoss > epsilon) {
            StringBuilder sb = new StringBuilder();
            double accLoss = 0;

            for (int i = 0; i < n; i++) {
                loss = feedForward(inputs[i], desiredOutputs[i]);
                accLoss += Math.abs(loss);
                backProp(momentumRate, learningRate);
            }

            e++;
            epochLoss = accLoss / n;

            sb.append("-------------Epoch: ").append(e).append(" Epoch Loss: ").append(epochLoss).append("-------------");
            System.out.println(sb);

            epochResult.append(e).append(',').append(epochLoss).append('\n');
        }
    }

    protected double feedForward(double[] inputVect, double[] desiredOutputVect) {
        double[][] inputMat = new double[inputVect.length][1];
        for (int i = 0; i < inputVect.length; i++) {
            inputMat[i][0] = inputVect[i];
        }

        activations[0] = new Matrix(inputMat);
        for (int i = 1; i < layerCount; i++) {
            MathFunction activationFn = i == layerCount - 1 ? outputLayerActivationFn : hiddenLayerActivationFn;

            Matrix net = Matrix.multiply(weights[i - 1], activations[i - 1]);
            activations[i] = Matrix.applyFunction(net, activationFn);
            nets[i] = net;
        }

        return calcLoss(desiredOutputVect);
    }

    // calculate Loss
    private double calcLoss(double[] desiredOutputVect) {
        double loss = 0;
        if (type == NETWORK_TYPE.REGRESSION) {
            // use mse
            double sse = 0;

            for (int i = 0; i < desiredOutputLength; i++) {
                double error = desiredOutputVect[i] - activations[layerCount - 1].data[i][0];
                sse = sse + (error * error);
                this.diffLossVect[i] = error;
            }
            loss = 0.5 * (sse / desiredOutputLength);

        } else if (type == NETWORK_TYPE.BIN_CLASSIFIER) {
            // use binary cross-entropy
            double y = desiredOutputVect[0];
            double p = activations[layerCount - 1].data[0][0];

            if (y == 0.0) {
                loss = -Math.log(1.0 - p);
                this.diffLossVect[0] = 1.0 / (1.0 - p);
            } else if (y == 1.0) {
                loss = -Math.log(p);
                this.diffLossVect[0] = -1.0 / p;
            }
        }
        return loss;
    }

    private void backProp(double momentumRate, double learningRate) {
        calcGrads();

        // update weights
        for (int l = 0; l < layerCount - 1; l++) {
            for (int j = 0; j < nodeInLayerCount[l + 1]; j++) {
                for (int i = 0; i < nodeInLayerCount[l]; i++) {
                    double momentumTerm = momentumRate * lastDeltaWeights[l].data[j][i];
                    double learningRateTerm = learningRate * grads[l + 1].data[j][0] * activations[l].data[i][0];
                    double deltaWeight = momentumTerm + learningRateTerm;

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
            grads[layerCount - 1].data[i][0] = diffLossVect[i] * diffOutputLayerActivationFn.run(nets[layerCount - 1]
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
                grads[l].data[j][0] = diffHiddenLayerActivationFn.run(nets[l].data[j][0]) * sumGradsWeight;
            }
        }
    }

    protected double[][] copy2Darray(double[][] src, int startCol, int colLen) {
        double[][] dataClone = new double[src.length][colLen];

        for (int i = 0; i < dataClone.length; i++) {
            double[] orgRow = src[i];
            System.arraycopy(orgRow, startCol, dataClone[i], 0, colLen);
        }
        return dataClone;
    }
}
