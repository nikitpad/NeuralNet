using System;

namespace NeuralLib.Maths
{
    public class Sigmoid
    {
        public static Func<double, double> Function => x => 1d / (1 + (double)Math.Exp(-x));

        public static Func<double, double> Derivative => x => Function(x) * (1 - Function(x));
    }
}
