using System;
using System.Diagnostics;
using System.Threading;
using lafd4net;
using MxNet;
using MxNet.Image;
using NumpyDotNet;
using OpenCvSharp;

namespace LAFDTest {
    class Program {
        static void Main(string[] args) {
            Console.WriteLine("Hello World!");
            bool debugSingle = false;
            bool debugAll = false;
            
            LFFD lffd = new LFFD("C:/machina/models/anime/symbol.json", "C:/machina/models/anime/model.params");
            var read = Cv2.ImRead(@"C:/machina/test/imgtest5.png").CvtColor(ColorConversionCodes.BGR2RGB);
            NDArray? ndarr;
            if (debugAll) {
                 ndarr = lffd.Predict(read, nmsFlag: false);
            }
            else {
                ndarr = lffd.Predict(read);
            }

            if (ndarr == null) {
                Console.WriteLine("No response!");
                return;
            }
            
            Console.WriteLine($"Found {ndarr.Shape[0]} bboxes.");

            var boxes = ndarr.AsNumpy();
            Cv2.NamedWindow("ShowDemo");
            Mat m = new Mat();

            if (debugSingle) {
                for (int i = 0; i < boxes.shape.iDims[0]; i++) {
                    float xmin = (float) ((ndarray) boxes[i])[0];
                    float ymin = (float) ((ndarray) boxes[i])[1];
                    float xmax = (float) ((ndarray) boxes[i])[2];
                    float ymax = (float) ((ndarray) boxes[i])[3];
                    float confidence = (float) ((ndarray) boxes[i])[4];
                    read.CopyTo(m);
                    m.Rectangle(new Point(xmin, ymin), new Point(xmax, ymax), Scalar.Red, 2);
                    m.PutText($"{confidence * 100}%", new Point(xmin, ymin), HersheyFonts.HersheySimplex, 0.8d,
                        Scalar.Orange, 2);
                    Cv2.ImShow("ShowDemo", m);
                    Cv2.WaitKey(0);
                }
                Cv2.DestroyWindow("ShowDemo");
                Environment.Exit(0);
            }
            else {
                read.CopyTo(m);
                foreach (ndarray box in boxes) {
                    float xmin = (float) box[0];
                    float ymin = (float) box[1];
                    float xmax = (float) box[2];
                    float ymax = (float) box[3];
                    float confidence = (float) box[4];
                    m.Rectangle(new Point(xmin, ymin), new Point(xmax, ymax), Scalar.Red, 2);
                    m.PutText($"{confidence * 100}%", new Point(xmin, ymin), HersheyFonts.HersheySimplex, 0.8d,
                        Scalar.Orange, 2);
                }
                Cv2.ImShow("ShowDemo", m);
            }

            //Cv2.ImWrite("output_dn.png", read);
            Cv2.WaitKey(0);
            Cv2.DestroyWindow("ShowDemo");
            Environment.Exit(0);
        }
    }
}