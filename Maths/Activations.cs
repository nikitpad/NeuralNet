using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralLib.Maths
{
    public class Activations
    {
        public enum Type
        {
            ReLU,
            LeakyReLU,
            Sigmoid
        }

        public static Func<double, double>[] Table = new Func<double, double>[]
        {
            ReLU.Function,
            ReLU.Leaky.Function,
            Sigmoid.Function
        };

        public static Func<double, double>[] TablePrime = new Func<double, double>[]
        {
            ReLU.Derivative,
            ReLU.Leaky.Derivative,
            Sigmoid.Function
        };

        public static Func<double, double> Get(Type t) => Table[(int)t];

        public static Func<double, double> GetPrime(Type t) => TablePrime[(int)t];
    }
}
