package com.Tanadol.MLP;

import java.util.Random;


public class Network {
    private int layerCount;
    private int[] nodeInLayerCount;
    private int nodeCount;

    private Matrix[] networkWeightsMat;

    private final double MIN_WEIGTH = -1.0;
    private final double MAX_WEIGHT = 1.0;

    private static final Random random = new Random();

    public Network(int[] nodeInLayerCount) {
        this.layerCount = nodeInLayerCount.length;
        this.nodeInLayerCount = new int[layerCount];
        for (int i = 0; i < nodeInLayerCount.length; i++) {
            this.nodeInLayerCount[i] = nodeInLayerCount[i];
            this.nodeCount += nodeInLayerCount[i];
        }

        networkWeightsMat = new Matrix[layerCount - 1];
        initWeight();
    }

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
}
