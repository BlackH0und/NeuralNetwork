package ch.marcschifferle.neuro;

import ch.marcschifferle.maths.Calculations;

import java.io.Serializable;
import java.util.ArrayList;

public class Brain implements Serializable {

    private final NeuralLayer layer1, layer2, layer3;
    private final NeuralLayer[] midLayers;
    private double[][] outputLayer1, outputLayer2, outputLayer3;
    private ArrayList<double[][]> outputLayers = new ArrayList<>();
    private ArrayList<NeuralLayer> neuralLayers = new ArrayList<>();
    private final double learningRate;

    public Brain(NeuralLayer layer1, NeuralLayer layer2, NeuralLayer layer3, NeuralLayer... midLayers) {
        this.layer1 = layer1;
        this.midLayers = midLayers;
        this.layer2 = layer2;
        this.layer3 = layer3;
        learningRate = .01;
    }
    public void think(double[][] inputs) {
        outputLayer1 = Calculations.apply(Calculations.matrixMultiply(inputs, layer1.weights), layer1.activationFunction);
        outputLayer2 = Calculations.apply(Calculations.matrixMultiply(outputLayer1, layer2.weights), layer2.activationFunction);

        outputLayer3 = Calculations.apply(Calculations.matrixMultiply(outputLayer2, layer3.weights), layer3.activationFunction);
    }

    public void train(double[][] inputs, double[][] outputs, int numberOfTrainingIterations) {
        for (int i = 0; i < numberOfTrainingIterations; ++i) {

            think(inputs);

            double[][] errorLayer3 = Calculations.matrixSubtract(outputs, outputLayer3);
            double[][] deltaLayer3 = Calculations.scalarMultiply(errorLayer3, Calculations.apply(outputLayer3, layer3.activationFunctionDerivative));

            double[][] errorLayer2 = Calculations.matrixMultiply(deltaLayer3, Calculations.matrixTranspose(layer3.weights));
            double[][] deltaLayer2 = Calculations.scalarMultiply(errorLayer2, Calculations.apply(outputLayer2, layer2.activationFunctionDerivative));

            double[][] errorLayer1 = Calculations.matrixMultiply(deltaLayer2, Calculations.matrixTranspose(layer2.weights));
            double[][] deltaLayer1 = Calculations.scalarMultiply(errorLayer1, Calculations.apply(outputLayer1, layer1.activationFunctionDerivative));

            double[][] adjustmentLayer1 = Calculations.matrixMultiply(Calculations.matrixTranspose(inputs), deltaLayer1);
            double[][] adjustmentLayer2 = Calculations.matrixMultiply(Calculations.matrixTranspose(outputLayer1), deltaLayer2);
            double[][] adjustmentLayer3 = Calculations.matrixMultiply(Calculations.matrixTranspose(outputLayer2), deltaLayer3);

            adjustmentLayer1 = Calculations.apply(adjustmentLayer1, (x) -> learningRate * x);
            adjustmentLayer2 = Calculations.apply(adjustmentLayer2, (x) -> learningRate * x);
            adjustmentLayer3 = Calculations.apply(adjustmentLayer3, (x) -> learningRate * x);

            this.layer1.adjustWeights(adjustmentLayer1);
            this.layer2.adjustWeights(adjustmentLayer2);
            this.layer3.adjustWeights(adjustmentLayer3);

            if (i % 10000 == 0) {
                System.out.println(" Training iteration " + i + " of " + numberOfTrainingIterations);
                //System.out.println(Calculations.matrixToString(this.layer1.weights));
            }
        }
    }

    public double[][] getOutput() {
        return outputLayer1;
    }

    public NeuralLayer getLayer1() {
        return layer1;
    }

    public NeuralLayer getLayer2() {
        return layer2;
    }

    public NeuralLayer getLayer3() {
        return layer3;
    }

    @Override
    public String toString() {
        String result = "Layer 1\n";
        result += layer1.toString();

        if (outputLayer1 != null) {
            result += "Layer 1 output\n";
            result += Calculations.matrixToString(outputLayer1);
        }

        return result;
    }
}
