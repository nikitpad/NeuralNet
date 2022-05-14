using System;

namespace NeuralLib.Maths
{
    public class GaussRandom
    {
        public static void RandomizeArray(MArray arr, double mean, double deviation, Random rand = null)
        {
            rand = rand ?? new Random();

            for (int i = 0; i < arr.Values.Length; i++)
            {
                double u1 = 1 - rand.NextDouble();
                double u2 = 1 - rand.NextDouble();
                double randStdNormal = Math.Sqrt(-2 * Math.Log(u1)) * Math.Sin(2 * Math.PI * u2);
                double randNormal = mean + deviation * randStdNormal;
                arr.Values[i] = (double)randNormal;
            }
        }
    }
}