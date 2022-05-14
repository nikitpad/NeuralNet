using System;

namespace NeuralLib.Maths
{
    public class ReLU
    {
        public class Leaky
        {
            public static Func<double, double> Function => x => x <= 0 ? .01 * x : x;

            public static Func<double, double> Derivative => x => x <= 0 ? .01 : 1;
        }

        public static Func<double, double> Function => x => x <= 0 ? 0 : x;

        public static Func<double, double> Derivative => x => x <= 0 ? 0 : 1;
    }
}
