package ch.marcschifferle.neuro;

import ch.marcschifferle.maths.Calculations;

import java.io.Serializable;
import java.util.function.Function;

public class NeuralLayer implements Serializable {

    public double[][] weights;

    public enum ActivationFunctionType {
        SIGMOID
    }

    public enum InitialWeightType {
        RANDOM
    }

    public final Function<Double, Double> activationFunction, activationFunctionDerivative;

    public NeuralLayer(int numberOfNeurons, int numberOfInputsPerNeuron) {
        this(ActivationFunctionType.SIGMOID, InitialWeightType.RANDOM, numberOfNeurons, numberOfInputsPerNeuron);
    }

    public NeuralLayer(ActivationFunctionType activationFunctionType, int numberOfNeurons, int numberOfInputsPerNeuron) {
        this(activationFunctionType, InitialWeightType.RANDOM, numberOfNeurons, numberOfInputsPerNeuron);
    }

    public NeuralLayer(ActivationFunctionType activationFunctionType, InitialWeightType initialWeightType, int numberOfNeurons, int numberOfInputsPerNeuron) {
        weights = new double[numberOfInputsPerNeuron][numberOfNeurons];

        for (int i = 0; i < numberOfInputsPerNeuron; ++i) {
            for (int j = 0; j < numberOfNeurons; ++j) {
                if (InitialWeightType.RANDOM == initialWeightType) {
                    weights[i][j] = (2 * Math.random()) - 1;
                }
            }
        }
        activationFunction = Calculations::sigmoid;
        activationFunctionDerivative = Calculations::sigmoidDerivative;

    }

    public void adjustWeights(double[][] adjustment) {
        this.weights = Calculations.matrixAdd(weights, adjustment);
    }

    @Override
    public String toString() {
        return Calculations.matrixToString(weights);
    }
}
