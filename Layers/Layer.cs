using NeuralLib.Maths;

namespace NeuralLib.Layers
{
    public enum LayerType
    {
        Conv,
        Dense,
        Pool
    }

    public abstract class Layer
    {
        public abstract LayerType Type { get; }

        public abstract int InputCount { get; set; }

        public abstract int OutputCount { get; set; }

        public abstract MArray Weights { get; set; }

        public abstract MArray Bias { get; set; }

        public abstract void Backward(
            MArray prevActivations, MArray thisLinear,
            ref MArray partialDelta, out MArray gWeights, out MArray gBias);

        public abstract MArray Forward(MArray input);

        public abstract MArray Linear(MArray input);
    }
}
