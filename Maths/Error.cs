using System;

namespace NeuralLib.Maths
{
    public class Error
    {
        public static double MeanSquares(MArray output, MArray expected)
        {
            double result = 0;

            for (int i = 0; i < output.Values.Length; i++)
            {
                double delta = output.Values[i] - expected.Values[i];
                result += delta * delta;
            }

            return result / 2;
        }

        public static double CrossEntropy(MArray output, MArray expected)
        {
            double E = 0;

            for (int i = 0; i < output.Values.Length; i++)
                E -= Math.Log(output.Values[i]) * expected.Values[i];

            return E;
        }
    }
}
