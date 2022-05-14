using NeuralLib.Layers;
using NeuralLib.Maths;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralLib
{

    public class Network
    {
        public List<Layer> Layers { get; set; } = new List<Layer>();

        public MArray Forward(MArray input)
        {
            MArray ffw = input;

            foreach (Layer layer in Layers)
                ffw = layer.Forward(ffw);

            return ffw;
        }

        public void StochasticGradientDescent(int times, double rate, int batchSize, (MArray, MArray)[] trainingExamples)
        {
            Random random = new Random();

            int t = 0;

            var weightGradientsM = new List<MArray>();
            var biasGradientsM = new List<MArray>();
            var weightGradientsV = new List<MArray>();
            var biasGradientsV = new List<MArray>();

            double b1 = .9, b2 = .999;
            double epsilon = 10e-4;

            foreach (Layer layer in Layers)
            {
                weightGradientsM.Add(layer.Weights == null ? null : new MArray(layer.Weights.Dimensions));
                biasGradientsM.Add(layer.Bias == null ? null : new MArray(layer.Bias.Dimensions));

                weightGradientsV.Add(layer.Weights == null ? null : new MArray(layer.Weights.Dimensions));
                biasGradientsV.Add(layer.Bias == null ? null : new MArray(layer.Bias.Dimensions));
            }

            for (int i = 0; i < times; i++)
            {
                int batches = trainingExamples.Length / batchSize;

                // Shuffle training examples
                for (int k = trainingExamples.Length - 1; 1 <= k; k--)
                {
                    int j = random.Next(0, k + 1);

                    var tmp = trainingExamples[j];
                    trainingExamples[j] = trainingExamples[k];
                    trainingExamples[k] = tmp;
                }

                for (int j = 0; j < batches; j++, t++)
                {
                    var weightGradients = new List<MArray>();
                    var biasGradients = new List<MArray>();

                    foreach (Layer layer in Layers)
                    {
                        weightGradients.Add(layer.Weights == null ? null : new MArray(layer.Weights.Dimensions));
                        biasGradients.Add(layer.Bias == null ? null : new MArray(layer.Bias.Dimensions));
                    }

                    double avgLoss = 0;

                    int correct = 0;
                    
                    object lck = new object();

                    Parallel.For(0, batchSize, k =>
                    {
                        int index = j * batchSize + k;

                        if (index < trainingExamples.Length)
                        {

                            (MArray, MArray) example = trainingExamples[index];

                            var linears = new List<MArray>();
                            var activations = new List<MArray>();
                            var weights = new List<MArray>();


                            MArray ffw = example.Item1;

                            activations.Add(ffw);

                            foreach (Layer layer in Layers)
                            {
                                linears.Add(layer.Linear(ffw));
                                activations.Add(ffw = layer.Forward(ffw));
                                weights.Add(layer.Weights);
                            }

                            activations[activations.Count - 1] = Softmax.Apply(activations.Last());

                            MArray partialDelta = activations.Last() - example.Item2;

                            //int ind = example.Item2.Values.ToList().IndexOf(1);

                            double loss = Error.CrossEntropy(activations.Last(), example.Item2);

                            lock (lck)
                                avgLoss += loss;

                            int correctIndex = example.Item2.Values.ToList().IndexOf(1d);
                            var valuesList = activations.Last().Values.ToList();

                            if (correctIndex == valuesList.IndexOf(valuesList.Max()))
                                lock (lck)
                                    correct++;

                            //Console.WriteLine($"batch #{j}: #{k}: {partialDelta.Values.Average()} {example.Item2} {activations.Last()}\n{Error.CrossEntropy(activations.Last(), example.Item2)}");
                            //Console.WriteLine($"batch #{j}: #{k}: {Error.CrossEntropy(activations.Last(), example.Item2)}");

                            Layers.Last().Backward(
                                activations[activations.Count - 2],
                                linears[linears.Count - 1],
                                ref partialDelta,
                                out MArray gWeights,
                                out MArray gBias);

                            if (gWeights != null)
                                lock (lck)
                                    weightGradients[Layers.Count - 1] += gWeights / batchSize;

                            if (gBias != null)
                                lock (lck)
                                    biasGradients[Layers.Count - 1] += gBias / batchSize;

                            for (int l = Layers.Count - 2; l >= 0; l--)
                            {
                                Layers[l].Backward(
                                    activations[l],
                                    linears[l],
                                    ref partialDelta,
                                    out gWeights,
                                    out gBias);

                                if (gWeights != null)
                                    lock (lck)
                                        weightGradients[l] += gWeights / batchSize;

                                if (gBias != null)
                                    lock (lck)
                                        biasGradients[l] += gBias / batchSize;
                            }
                        }
                    });

                    avgLoss /= batchSize;
                    Console.WriteLine($"Epoch {i}, batch {j}, average loss = {avgLoss}, accuracy = {correct / (double)batchSize * 100}%;");

                    //double avg = 0;

                    double f1 = 1d / (1 - Math.Pow(b1, t));
                    double f2 = 1d / (1 - Math.Pow(b2, t));

                    for (int k = 0; k < Layers.Count; k++)
                    {
                        if (Layers[k].Weights != null)
                        {
                            //avg += (weightGradients[k] * (rate / batchSize)).Values.Average();
                            weightGradientsM[k] = weightGradientsM[k] * b1 + weightGradients[k] * (1 - b1);
                            weightGradientsV[k] = weightGradientsV[k] * b2 + weightGradients[k].ApplyFunc(x => x * x) * (1 - b2);

                            Layers[k].Weights -= 
                                weightGradientsM[k] / f1
                                / (weightGradientsV[k] / f2).ApplyFunc(x => Math.Sqrt(x) + epsilon) * rate;;
                        }

                        if (Layers[k].Bias != null)
                        {
                            biasGradientsM[k] = biasGradientsM[k] * b1 + biasGradients[k] * (1 - b1);
                            biasGradientsV[k] = biasGradientsV[k] * b2 + biasGradients[k].ApplyFunc(x => x * x) * (1 - b2);

                            Layers[k].Bias -= 
                                biasGradientsM[k] / f1
                                / (biasGradientsV[k] / f2).ApplyFunc(x => Math.Sqrt(x) + epsilon) * rate;
                        }
                    }

                    //avg /= Layers.Count;
                    //Debug.WriteLine($"#{i} Avg gradient: {avg}");
                }
            }
        }
    }
}