package ch.marcschifferle.maths;

import java.util.function.Function;

public class Calculations {

    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double sigmoidDerivative(double x) {
        return x * (1 - x);
    }


    public static double[][] scalarMultiply(double[][] v1, double[][] v2) {
        if (v1.length != v2.length) {
            throw new IllegalArgumentException("scalar Multiplication error");
        }
        double result[][] = new double[v1.length][v1[0].length];
        for (int i = 0; i < v1.length; ++i) {
            for (int j = 0; j < v1[i].length; ++j) {
                result[i][j] = v1[i][j] * v2[i][j];
            }
        }
        return result;

    }
    public static double[][] matrixMultiply(double[][] a, double[][] b) {
        if (a.length == 0 || b.length == 0 || a[0].length != b.length) {
            throw new IllegalArgumentException("matrix multiplication error");
        }

        int n = a.length;
        int m = a[0].length;
        int p = b[0].length;

        double[][] result = new double[n][p];

        for (int nIter = 0; nIter < n; ++nIter) {
            for (int pIter = 0; pIter < p; ++pIter) {

                double sum = 0;
                for (int mIter = 0; mIter < m; ++mIter) {
                    sum += (a[nIter][mIter] * b[mIter][pIter]);
                }

                result[nIter][pIter] = sum;
            }
        }

        return result;
    }

    public static double[][] matrixAdd(double[][] a, double[][] b) {
        if (a.length == 0 || b.length == 0 || a.length != b.length || a[0].length != b[0].length) {
            throw new IllegalArgumentException("matrix addition error");
        }

        double result[][] = new double[a.length][a[0].length];

        for (int i = 0; i < a.length; ++i) {
            for (int j = 0; j < a[i].length; ++j) {
                result[i][j] = a[i][j] + b[i][j];
            }
        }

        return result;
    }

    public static double[][] matrixSubtract(double[][] a, double[][] b) {
        if (a.length == 0 || b.length == 0 || a.length != b.length || a[0].length != b[0].length) {
            throw new IllegalArgumentException("matrix subtraction error");
        }

        double result[][] = new double[a.length][a[0].length];

        for (int i = 0; i < a.length; ++i) {
            for (int j = 0; j < a[i].length; ++j) {
                result[i][j] = a[i][j] - b[i][j];
            }
        }

        return result;
    }

    public static double[][] matrixTranspose(double[][] matrix) {
        double[][] result = new double[matrix[0].length][matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; ++j) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }

    public static double[][] apply(double[][] matrix, Function<Double, Double> fn) {
        if (matrix.length == 0 || matrix[0].length == 0) {
            throw new IllegalArgumentException("matrix is empty");
        }

        double[][] result = new double[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; ++i) {
            for (int j = 0; j < matrix[i].length; ++j) {
                result[i][j] = fn.apply(matrix[i][j]);
            }
        }

        return result;
    }

    public static String matrixToString(double[][] matrix) {
        String result = "[";
        for (double[] aMatrix : matrix) {
            result += "[";
            for (double anAMatrix : aMatrix) {
                result += anAMatrix + " ";
            }
            result += "]\n";
        }
        result += "]\n";

        return result;
    }
}
