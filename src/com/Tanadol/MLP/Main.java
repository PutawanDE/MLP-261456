package com.Tanadol.MLP;

public class Main {
    public static void main(String[] args) {
        int[] nodeInLayerCount = new int[]{4, 2, 1};
        Network network = new Network(nodeInLayerCount);
        network.train(new double[][]{{20, 10, 10, 0}, {10.2, 2.0, 5.0, 0.5}}, 1, 1);
        System.out.println();
    }
}
