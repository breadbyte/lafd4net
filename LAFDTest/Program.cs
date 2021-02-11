using System;
using System.Diagnostics;
using System.Drawing.Imaging;
using System.IO;
using System.Threading;
using lafd4net;
using MxNet;
using MxNet.Image;
using NumpyDotNet;
using OpenCvSharp;
using Image = System.Drawing.Image;

namespace LAFDTest {
    class Program {
        static void Main(string[] args) {
            Console.WriteLine("Hello World!");
            LFFD lffd = new LFFD("C:/machina/models/anime/symbol.json", "C:/machina/models/anime/model.params");
            var read = Image.FromFile(@"C:/machina/test/imgtest4.png");
            var resizeScale = Math.Min(384f / Math.Max(read.Width, read.Height), 1f);
            Cv2.NamedWindow("ShowDemo");

            bool useSystemDrawing = true;
            if (useSystemDrawing) {

                var result = lffd.PredictWholeImage(read, resizeScale);
                if (result == null) {
                    Console.WriteLine("Prediction did not return anything!");
                    Environment.Exit(1);
                }

                MemoryStream memStream = new MemoryStream();
                result.Save(memStream, ImageFormat.Png);
                
                Cv2.ImShow("ShowDemo", Cv2.ImDecode(memStream.ToArray(), ImreadModes.Color));
            }

            else {
                var memStream = new MemoryStream();
                read.Save(memStream, read.RawFormat);
                var readCv2 = Cv2.ImDecode(memStream.ToArray(), ImreadModes.Color);
                NDArray? ndarr;

                ndarr = lffd.Predict(readCv2, resizeScale);
                if (ndarr == null) {
                    Console.WriteLine("Prediction did not return anything!");
                    return;
                }

                Console.WriteLine($"Found {ndarr.Shape[0]} bboxes.");

                var boxes = ndarr.AsNumpy();
                Mat m = new Mat();
                readCv2.CopyTo(m);
                m = m.CvtColor(ColorConversionCodes.BGR2RGB);
                foreach (ndarray box in boxes) {
                    float xmin = (float) box[0];
                    float ymin = (float) box[1];
                    float xmax = (float) box[2];
                    float ymax = (float) box[3];
                    float confidence = (float) box[4];
                    m.Rectangle(new Point(xmin, ymin), new Point(xmax, ymax), Scalar.Red, 2);
                    var shape = Cv2.GetTextSize($"{confidence * 100}%", HersheyFonts.HersheySimplex, 1d, 2,
                        out var baseline);
                    m.Rectangle(new Point(xmin, ymax - shape.Height - baseline),
                        new Point(xmin + shape.Width, ymax), Scalar.Red, -1);
                    m.PutText($"{confidence * 100}%", new Point(xmin, ymax), HersheyFonts.HersheySimplex, 1d,
                        Scalar.Orange, 2);
                }
                Cv2.ImShow("ShowDemo", m);
            }


            Cv2.WaitKey(0);
            Cv2.DestroyWindow("ShowDemo");
            Environment.Exit(0);
        }
    }
}