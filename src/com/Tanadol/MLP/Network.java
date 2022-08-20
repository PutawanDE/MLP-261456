package com.Tanadol.MLP;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

interface MathFunction {
    double run(double x);
}
/* TODO: -use same arrays index as the notation in http://myweb.cmu.ac.th/sansanee.a/IntroCI/Review_1st_Midterm.pdf slide
    - compute error in feed forward instead of in backprop
    - use feed forward to evaluate input
    - more generalized code to be easily used by another data set
    - refactor error calculation
    - Think about epsilon
*/
public class Network {
    private int layerCount;
    private int[] nodeInLayerCount;

    private double max;
    private double min;

    private Matrix[] activations;
    private Matrix[] weights;
    private Matrix[] grads;
    private Matrix[] nets;
    private Matrix[] lastDeltaWeights;
    private double[] error;

    private double[] desiredOutput;
    private double sse;
    private double sse_water;
    private double mse = 1;
    private double mse_water;
    private StringBuilder stringBuilder = new StringBuilder();

    private final double MIN_WEIGTH = -1.0;
    private final double MAX_WEIGHT = 1.0;

    // f(x)=max(0, x)
    private final MathFunction reluFn = (x) -> {
        if (x < 0) return 0;
        else return x;
    };

    private final MathFunction leakyReluFn = (x) -> {
        if (x <= 0) return 0.01 * x;
        else return x;
    };

    private final MathFunction diffReluFn = (x) -> {
        if (x <= 0) return 0;
        else return 1;
    };

    private final MathFunction diffLeakyReluFn = (x) -> {
        if (x <= 0) return 0.01;
        else return 1;
    };

    private static final Random random = new Random();

    public Network(int[] nodeInLayerCount) {
        this.layerCount = nodeInLayerCount.length;
        this.nodeInLayerCount = nodeInLayerCount;

        weights = new Matrix[layerCount - 1];
        lastDeltaWeights = new Matrix[layerCount - 1];
        activations = new Matrix[layerCount];
        grads = new Matrix[layerCount];
        nets = new Matrix[layerCount];

        initWeight();
    }

    public void train(double[][] dataset, double momentumRate, double learningRate, int maxEpoch, String name) {
        double[][] dataClone = new double[dataset.length][dataset[0].length];
        for (int i = 0; i < dataClone.length; i++) {
            double[] orgRow = dataset[i];
            dataClone[i] = new double[orgRow.length];
            System.arraycopy(orgRow, 0, dataClone[i], 0, orgRow.length);
        }

        min_max_norm(dataClone);
        desiredOutput = new double[1];
        int n = dataClone.length;

        int e = 0;
        stringBuilder.append("Epoch,").append("MSE\n");
        while (e < maxEpoch) { // while
            for (int i = 0; i < dataClone.length; i++) {
                feedForward(dataClone[i]);
                desiredOutput[0] = dataClone[i][dataClone[i].length - 1];
                backProp(momentumRate, learningRate);
            }
            e++;
            mse = sse / n;
            mse_water = mse_water / n;
            sse_water = 0;
            sse = 0;
            stringBuilder.append("--------------Epoch: ").append(e).append(" MSE: ").append(mse)
                    .append("-----------------\n");
            stringBuilder.append(e).append(',').append(mse).append('\n');
        }
        System.out.println(stringBuilder);
//        saveResult(name);
        stringBuilder = new StringBuilder();
    }

    public double evaluateInput(double[][] dataset, StringBuilder evalStringSb) {
        double[][] dataClone = new double[dataset.length][dataset[0].length];
        for (int i = 0; i < dataClone.length; i++) {
            double[] orgRow = dataset[i];
            dataClone[i] = new double[orgRow.length];
            System.arraycopy(orgRow, 0, dataClone[i], 0, orgRow.length);
        }

        sse = 0;
        mse = 0;
        min_max_norm(dataClone);

        for (int i = 0; i < dataClone.length; i++) {
            feedForward(dataClone[i]);
            desiredOutput[0] = dataClone[i][dataClone[i].length - 1];
            calcErrors();
        }
        mse = sse / dataClone.length;
        double rmse = Math.sqrt(mse);
        mse_water = sse_water / dataClone.length;
        double rmse_water = Math.sqrt(mse_water);
        sse_water = 0;
        sse = 0;
        evalStringSb.append(rmse).append(',').append(rmse_water).append(',');
//        saveResult(name);
//        System.out.println(evalStringSb);
        return rmse;
    }

    public void saveResult(String name) {
        File file = new File("D:/PUTAWAN/ComputerProjects/CI/" + name + ".csv");
        try (BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(file))) {
            bufferedWriter.append(stringBuilder);
        } catch (IOException exception) {
            exception.printStackTrace();
        }
    }

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
    // normalize only input
    private void min_max_norm(double[][] dataset) {
        max = Integer.MIN_VALUE;
        min = Integer.MAX_VALUE;

        for (int i = 0; i < dataset.length; i++) {
            for (int j = 0; j < dataset[i].length - 1; j++) {
                if (dataset[i][j] > max) max = dataset[i][j];
                if (dataset[i][j] < min) min = dataset[i][j];
            }
        }

        double range = max - min;
        for (int i = 0; i < dataset.length; i++) {
            for (int j = 0; j < dataset[i].length - 1; j++) {
                dataset[i][j] = (dataset[i][j] - min) / range;
            }
        }
    }

    private void feedForward(double[] inputs) {
        double[][] inputVect = new double[inputs.length - 1][1];
        for (int i = 0; i < inputs.length - 1; i++) {
            inputVect[i][0] = inputs[i];
        }
        activations[0] = new Matrix(inputVect);
        for (int i = 1; i < layerCount; i++) {
            Matrix net = Matrix.multiply(weights[i - 1], (activations[i - 1]));
            activations[i] = Matrix.applyFunction(net, leakyReluFn);
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
            grads[layerCount - 1].data[i][0] = error[i] * diffLeakyReluFn.run(nets[layerCount - 1]
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
                grads[l].data[j][0] = diffLeakyReluFn.run(nets[l].data[j][0]) * sumGradsWeight;
            }
        }
    }

    // calculate SSE and output layer error
    private void calcErrors() {
        error = new double[desiredOutput.length];
        for (int i = 0; i < desiredOutput.length; i++) {
            double normDesireOutput = (desiredOutput[i] - min) / (max - min);
            double rawOutput = activations[layerCount - 1].data[i][0];
            double denormOutput = activations[layerCount - 1].data[i][0] * (max - min) + min;

            double error = normDesireOutput - activations[layerCount - 1].data[i][0];
            double denormError = desiredOutput[i] - denormOutput;
            sse_water = sse_water + (denormError * denormError);
            sse = sse + (error * error);
            this.error[i] = error;

//            stringBuilder.append("DesireOutput: ").append(desiredOutput[i]).append(" Raw Output: ").append(rawOutput)
//                    .append(" Output: ").append(denormOutput).append(" Error: ").append(error)
//                    .append(" Water Lv. Error: ").append(denormError).append("\n");
        }
    }
}
