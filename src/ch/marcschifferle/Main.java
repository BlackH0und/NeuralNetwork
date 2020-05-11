package ch.marcschifferle;


import ch.marcschifferle.maths.Calculations;
import ch.marcschifferle.neuro.Brain;
import ch.marcschifferle.neuro.NeuralLayer;

import java.io.*;

/**
 * Made by Marc Schifferle
 */

public class Main {


    public static void predict(double[][] testInput, double expected, Brain brain) {
        brain.think(testInput);

        System.out.println("Prediction on data "
                + testInput[0][0] + "  "
                + testInput[0][1] + "  "
                + testInput[0][2] + "  "
                + brain.getOutput()[0][0] + ", expected -> " + expected);
        System.out.println("\tDeviation -> " + (expected - brain.getOutput()[0][0]));
    }

    public static void main(String[] args) {
        File f = new File("layer1.nn");
        if (!f.exists()) {
            try {
                f.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        File f2 = new File("layer2.nn");
        if (!f2.exists()) {
            try {
                f2.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        File f3 = new File("layer3.nn");
        if (!f3.exists()) {
            try {
                f3.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        NeuralLayer layer1 = new NeuralLayer(1, 3);
        NeuralLayer layer2 = new NeuralLayer(3, 1);
        NeuralLayer layer3 = new NeuralLayer(1, 3);

        double[][] inputs = new double[][]{
                {0, 0, 1},
                {1, 1, 1},
                {1, 0, 1},
                {0, 1, 1}
        };

        double[][] outputs = new double[][]{
                {0},
                {1},
                {1},
                {0}
        };

        Brain brain = new Brain(layer1, layer2, layer3);
        for (int i = 0; i < 1; i++) {
            brain.train(inputs, outputs, 5000000);

            predict(new double[][]{{1, 0, 1}}, 1, brain);

            predict(new double[][]{{0, 1, 1}}, 0, brain);
        }

        /*try {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(f));
            double[][] d = (double[][]) ois.readObject();
            layer1.weights = d;
            ois.close();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        try {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(f2));
            double[][] d = (double[][]) ois.readObject();
            layer2.weights = d;
            ois.close();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        try {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(f3));
            double[][] d = (double[][]) ois.readObject();
            layer3.weights = d;
            ois.close();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        Brain brain = new Brain(layer1, layer2, layer3);


        System.out.println("Starting Training...");
        for (int x = 0; x < 10; x++) {


            int amountOfTrainsets = 1000;
            double[][] inputs = new double[amountOfTrainsets][2];
            double[][] outputs = new double[amountOfTrainsets][1];
            for (int i = 0; i < amountOfTrainsets; i++) {
                for (int j = 0; j < 2; j++) {
                    inputs[i][j] = (2 * Math.random()) - 1;
                }
                outputs[i][0] = inputs[i][0] + inputs[i][1];
                if (outputs[i][0] > 1 || outputs[i][0] < -1) {
                    i--;
                }
            }
            brain.train(inputs, outputs, 25000);

            double[][] testings = new double[2][2];
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    testings[i][j] = (2 * Math.random()) - 1;
                }
                predict(new double[][]{{inputs[i][0], inputs[i][1]}}, (inputs[i][0] + inputs[i][1]), brain);
            }


            System.out.println("Saving Data...");
            try {

                ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(f));
                oos.writeObject(brain.getLayer1().weights);
            } catch (IOException e) {
                e.printStackTrace();
            }
            try {
                ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(f2));
                oos.writeObject(brain.getLayer2().weights);
            } catch (IOException e) {
                e.printStackTrace();
            }
            try {
                ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(f3));
                oos.writeObject(brain.getLayer3().weights);
            } catch (IOException e) {
                e.printStackTrace();
            }

            System.out.println("Data saved");

        }
        System.out.println("Finished training");*/

    }
}
