package com.Tanadol.MLP;

public class Main {
    public static void main(String[] args) {
        int[] nodeInLayerCount = new int[]{4, 3, 2};
        Network network = new Network(nodeInLayerCount);
        network.train(new double[][]{{5.0, 1.0, 2.0, 3.0}, {10.2, 2.0, 5.0, 0.5}});
        System.out.println();
    }
}
