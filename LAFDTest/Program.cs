using System;
using System.Diagnostics;
using System.Threading;
using lafd4net;
using MxNet.Image;
using NumpyDotNet;
using OpenCvSharp;

namespace LAFDTest {
    class Program {
        static void Main(string[] args) {
            Console.WriteLine("Hello World!");
            LFFD lffd = new LFFD("C:/machina/models/anime/symbol.json", "C:/machina/models/anime/model.params");
            var read = Cv2.ImRead(@"C:/machina/test/imgtest2.png").CvtColor(ColorConversionCodes.BGR2RGB);
            var ndarr = lffd.Predict(read);
            if (ndarr == null) {
                Console.WriteLine("No response!");
                return;
            }

            var boxes = ndarr.AsNumpy();
            
            foreach (ndarray box in boxes) {
                float xmin = (float)box[0];
                float ymin = (float)box[1];
                float xmax = (float)box[2];
                float ymax = (float)box[3];
                float confidence = (float)box[4];
                Cv2.Rectangle(read, new Point(xmin, ymin), new Point(xmax, ymax), Scalar.Red, 2);
                Cv2.PutText(read, $"{confidence * 100}%", new Point(xmin, ymin), HersheyFonts.HersheySimplex, 0.8d, Scalar.Orange, 2);
            }
            
            Cv2.NamedWindow("ShowDemo");
            Cv2.ImShow("ShowDemo", read);

            Cv2.ImWrite("output_dn.png", read);
            Cv2.WaitKey();
            Environment.Exit(0);
        }
    }
}