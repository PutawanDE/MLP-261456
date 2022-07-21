package com.Tanadol.MLP;

import java.util.Arrays;
import java.util.stream.Collectors;

public class Matrix {
    private int rows;
    private int cols;

    public double[][] data;

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        data = new double[rows][cols];
    }

    public Matrix(double[][] data) {
        this.data = data.clone();
        this.rows = data.length;
        this.cols = data[0].length;
    }

    public int getRows() {
        return rows;
    }

    public int getCols() {
        return cols;
    }

    public Matrix add(Matrix matrix) {
        if (matrix.rows != rows && matrix.cols != cols) {
            throw new ArithmeticException("Matrix Addition is not possible.");
        }

        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = data[j][i] + matrix.data[j][i];
            }
        }

        return new Matrix(result);
    }

    public Matrix multiplyBy(Matrix matrix) {
        if (matrix.rows != cols) {
            throw new ArithmeticException("Matrix Multiplication is not possible.");
        }

        double[][] result = new double[this.rows][matrix.cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < matrix.cols; j++) {
                for (int k = 0; k < matrix.rows; k++) {
                    result[i][j] += data[i][k] * matrix.data[k][j];
                }
            }
        }
        return new Matrix(result);
    }

    @Override
    public String toString() {
        return Arrays
                .stream(data)
                .map(Arrays::toString)
                .collect(Collectors.joining(System.lineSeparator()));
    }
}
