using NeuralLib.Maths;
using System;

namespace NeuralLib.Layers
{
    public class DenseLayer : Layer
    {
        public override LayerType Type => LayerType.Dense;

        public DenseLayer() { }

        public DenseLayer(int inputCount, int outputCount,
            Activations.Type activation,
            bool randomInit, double initScale)
        {
            InputCount = inputCount;
            OutputCount = outputCount;
            Weights = new MArray(inputCount, outputCount);
            Bias = new MArray(outputCount);
            Activation = activation;

            if (randomInit)
            {
                double w = Math.Sqrt(6d / (inputCount + outputCount));
                //GaussRandom.RandomizeArray(Weights, 0, 1);
                Weights.UniformRandomize(-w, w);
                Weights *= initScale;
            }
        }

        public override int InputCount { get; set; }

        public override int OutputCount { get; set; }

        public override MArray Weights { get; set; }

        public override MArray Bias { get; set; }

        public Activations.Type Activation { get; set; }

        public override MArray Forward(MArray input)
            => Linear(input).ApplyFunc(Activations.Get(Activation));

        public override MArray Linear(MArray input) => Weights * input + Bias;

        public override void Backward(
            MArray prevActivations, MArray thisLinear,
            ref MArray partialDelta, out MArray gWeights, out MArray gBias)
        {
            //gBias = newPartialDelta = (layerCostPrime ?? nextWeights.T * delta) * thisLinear.ApplyFunc(ActivationPrime);
            gBias = partialDelta * thisLinear.ApplyFunc(Activations.GetPrime(Activation));
            gWeights = gBias * prevActivations.T;
            partialDelta = Weights.T * gBias;
        }
    }
}