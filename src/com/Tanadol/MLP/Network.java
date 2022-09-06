package com.Tanadol.MLP;

import java.util.Random;

interface MathFunction {
    double run(double x);
}

public class Network {
    protected final int inputLength;
    protected final int desiredOutputLength;

    protected int layerCount;
    private int[] nodeInLayerCount;

    protected Matrix[] activations;
    private Matrix[] weights;
    private Matrix[] grads;
    private Matrix[] nets;
    private Matrix[] lastDeltaWeights;
    private double[] errorVect;

    private final double minWeight;
    private final double maxWeight;

    private final MathFunction activationFn;
    private final MathFunction diffActivationFn;

    private static final Random random = new Random();

    public Network(int[] nodeInLayerCount, MathFunction activationFn, MathFunction diffActivationFn,
                   double minWeight, double maxWeight) {
        this.layerCount = nodeInLayerCount.length;
        this.nodeInLayerCount = nodeInLayerCount;
        this.activationFn = activationFn;
        this.diffActivationFn = diffActivationFn;
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
                             StringBuilder mseResult) {
        int e = 0;
        double sse = 0;
        double mse;
        double rmse = Integer.MAX_VALUE;

        int n = inputs.length;

        while (e < maxEpoch && rmse > epsilon) {
            StringBuilder sb = new StringBuilder();

            for (int i = 0; i < n; i++) {
                sse += feedForward(inputs[i], desiredOutputs[i]);
                backProp(momentumRate, learningRate);
            }

            e++;
            mse = sse / n;
            sse = 0;
            rmse = Math.sqrt(mse);

            sb.append("-------------Epoch: ").append(e).append(" MSE: ").append(mse).append("-------------");
            System.out.println(sb);

            mseResult.append(e).append(',').append(mse).append('\n');
        }
    }

    protected double feedForward(double[] inputVect, double[] desiredOutputVect) {
        double[][] inputMat = new double[inputVect.length][1];
        for (int i = 0; i < inputVect.length; i++) {
            inputMat[i][0] = inputVect[i];
        }
        activations[0] = new Matrix(inputMat);
        for (int i = 1; i < layerCount; i++) {
            Matrix net = Matrix.multiply(weights[i - 1], (activations[i - 1]));
            activations[i] = Matrix.applyFunction(net, activationFn);
            nets[i] = net;
        }

        return calcErrors(desiredOutputVect);
    }

    // calculate SSE and output layer error
    private double calcErrors(double[] desiredOutputVect) {
        double sse = 0;
        errorVect = new double[desiredOutputVect.length];

        for (int i = 0; i < desiredOutputVect.length; i++) {
            double error = desiredOutputVect[i] - activations[layerCount - 1].data[i][0];
            sse = sse + (error * error);
            this.errorVect[i] = error;
        }
        return sse;
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
            grads[layerCount - 1].data[i][0] = errorVect[i] * diffActivationFn.run(nets[layerCount - 1]
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
                grads[l].data[j][0] = diffActivationFn.run(nets[l].data[j][0]) * sumGradsWeight;
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
