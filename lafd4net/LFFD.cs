using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using MxNet;
using MxNet.Image;
using MxNet.IO;
using MxNet.Modules;
using NumpyDotNet;
using NumpyLib;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using Point = OpenCvSharp.Point;
using Size = OpenCvSharp.Size;

namespace lafd4net {
    public class LFFD {
        private int[] _bboxSmallList = new[] {10, 20, 40, 80, 160};
        private int[] _bboxLargeList = new[] {20, 40, 80, 160, 320};
        private int[] _receptiveFieldList = new[] {20, 40, 80, 160, 320};
        private int[] _receptiveFieldStride = new[] {4, 8, 16, 32, 64};
        private int[] _receptiveFieldCenterStart = new[] {3, 7, 15, 31, 63};
        private string symbolPath;
        private string modelPath;
        private List<float> constant = new();
        private int inputHeight = 480;
        private int inputWidth = 640;
        private int outputScales = 5;
        private bool alternativeInput = false;
        private string symbolJson;
        private byte[] modelBytes;

        private Module _mxNetModule;
        private Context _mxNetContext;

        public LFFD(string symbolPath, string modelPath) {
            this.symbolPath = symbolPath;
            this.modelPath = modelPath;

            foreach (var receptiveField in _receptiveFieldList) {
                constant.Add(receptiveField / 2);
            }

            _mxNetContext = Context.Cpu();
        }

        public LFFD(string symbolJson, byte[] modelFile) {
            alternativeInput = true;

            this.symbolJson = symbolJson;
            modelBytes = modelFile;
            
            foreach (var receptiveField in _receptiveFieldList) {
                constant.Add(receptiveField / 2);
            }

            _mxNetContext = Context.Cpu();
        }

        /// <summary>
        /// Returns an entire image with bounding boxes and confidence percentage.
        /// </summary>
        /// <param name="image"></param>
        /// <param name="resizeScale"></param>
        /// <param name="scoreThreshold"></param>
        /// <param name="topK"></param>
        /// <param name="nmsThreshold"></param>
        /// <param name="nmsFlag"></param>
        /// <returns></returns>
        public Image? PredictWholeImage(Image image, float resizeScale = 1f, float scoreThreshold = 0.7f, int topK = 10000, float nmsThreshold = 0.3f, bool nmsFlag = true) {
            var memStream = new MemoryStream();
            image.Save(memStream, image.RawFormat);
            var mat = Cv2.ImDecode(memStream.ToArray(), ImreadModes.Color);
            var matFinal = new Mat();
            var ndArray = Predict(mat, resizeScale, scoreThreshold, topK, nmsThreshold, nmsFlag);

            if (ndArray == null)
                return null;

            var boxes = ndArray.AsNumpy();
            matFinal = mat.CvtColor(ColorConversionCodes.BGR2RGB);
            foreach (ndarray box in boxes) {
                float xmin = (float) box[0];
                float ymin = (float) box[1];
                float xmax = (float) box[2];
                float ymax = (float) box[3];
                float confidence = (float) box[4];
                matFinal.Rectangle(new Point(xmin, ymin), new Point(xmax, ymax), Scalar.Red, 2);
                var shape = Cv2.GetTextSize($"{confidence * 100}%", HersheyFonts.HersheySimplex, 1d, 2, out var baseline);
                matFinal.Rectangle(new Point(xmin, ymax - shape.Height - baseline), new Point(xmin + shape.Width, ymax), Scalar.Red, -1);
                matFinal.PutText($"{confidence * 100}%", new Point(xmin, ymax), HersheyFonts.HersheySimplex, 1d, Scalar.Orange, 2);
            }

            return Image.FromStream(matFinal.ToMemoryStream());
        }
        
        /// <summary>
        /// Returns an N amount of detected faces.
        /// </summary>
        /// <param name="image"></param>
        /// <param name="resizeScale"></param>
        /// <param name="scoreThreshold"></param>
        /// <param name="topK"></param>
        /// <param name="nmsThreshold"></param>
        /// <param name="nmsFlag"></param>
        /// <returns></returns>
        public IEnumerable<Image> PredictFaces(Image image, float resizeScale = 1f, float scoreThreshold = 0.7f, int topK = 10000, float nmsThreshold = 0.3f, bool nmsFlag = true) {
            var memStream = new MemoryStream();
            image.Save(memStream, image.RawFormat);
            var mat = Cv2.ImDecode(memStream.ToArray(), ImreadModes.Color);
            var matFinal = new Mat();
            var ndArray = Predict(mat, resizeScale, scoreThreshold, topK, nmsThreshold, nmsFlag);

            if (ndArray == null)
                yield break;

            var boxes = ndArray.AsNumpy();
            matFinal = mat.CvtColor(ColorConversionCodes.BGR2RGB);
            foreach (ndarray box in boxes) {
                var xmin = (int)(float) box[0];
                var ymin = (int)(float) box[1];
                var xmax = (int)(float) box[2];
                var ymax = (int)(float) box[3];
                float confidence = (float) box[4];
                
                yield return Image.FromStream(new Mat(matFinal, Rect.FromLTRB(xmin, ymin, xmax, ymax)).ToMemoryStream());
            }
        }

        public NDArray? Predict(Mat image, float resizeScale = 1f, float scoreThreshold = 0.7f, int topK = 10000, float nmsThreshold = 0.3f, bool nmsFlag = true) {
            #if DEBUG
            Stopwatch s = new Stopwatch();
            s.Start();
            #endif

            // Load the image.
            var img = NDArray.LoadCV2Mat(image, _mxNetContext);
            
            // Invalid image?
            if (img.Dimension != 3 || img.Shape[2] != 3) {
                throw new InvalidDataException("Invalid image format");
            }

            // Resize our image.
            float shorterSide = Math.Min(image.Width, image.Height);

            if (shorterSide * resizeScale < 128f)
                resizeScale = 128f / shorterSide;

            if (resizeScale != 1f) {
                img = NDArray.LoadCV2Mat(image.Resize(Size.Zero, resizeScale, resizeScale));
            }

            // Prepare the image.
            var imgH = img.Shape[0];
            var imgW = img.Shape[1];
            img = img.AsType(DType.Float32);
            img = img.ExpandDims(-1);
            img = img.Transpose(new Shape(3, 2, 0, 1));

            
            // Load our model.
            #region load model

            var dataName = "data";
            var dataShape = new Shape(1, 3, imgH, imgW);
            var dataNameShape = new DataDesc(dataName, dataShape);

            _mxNetModule = new Module(alternativeInput ? Symbol.LoadJSON(symbolJson) : Symbol.Load(symbolPath), new[] {dataName}, null, new[] {_mxNetContext});
            _mxNetModule.Bind(new[] {dataNameShape}, for_training: false);

            var ndarrModel = alternativeInput ? NDArrayDict.LoadFromBuffer(modelBytes) : NDArray.Load(modelPath);
            var argNameArrays = new NDArrayDict();
            var auxNameArrays = new NDArrayDict();
            argNameArrays.Add("data", nd.Zeros(dataShape, _mxNetContext));

            foreach (var item in ndarrModel) {
                if (item.Key.StartsWith("arg:"))
                    argNameArrays.Add(item.Key.Replace("arg:", "").Trim(), item.Value.AsInContext(_mxNetContext));
                if (item.Key.StartsWith("aux:"))
                    auxNameArrays.Add(item.Key.Replace("aux:", "").Trim(), item.Value.AsInContext(_mxNetContext));
            }

            _mxNetModule.InitParams(null, argNameArrays, auxNameArrays, true);

            #endregion

            #if DEBUG
            Console.WriteLine($"Model loading took {s.ElapsedMilliseconds}ms.");
            s.Reset();
            s.Start();
            #endif

            // Give the data to MXNet.
            DataBatch batch = new DataBatch(img);
            _mxNetModule.Forward(batch, false);
            var results = _mxNetModule.GetOutputs();

            List<Tuple<float, BoundingBoxParams>> bboxCollection = new();

            // For all our outputs, filter it and add it to bboxCollection.
            for (int i = 0; i < outputScales; i++) {
                var scoreMap = nd.Squeeze(results[0][i * 2], new Shape(0, 1));
                var bboxMap = nd.Squeeze(results[0][i * 2 + 1], new Shape(0));

                List<float> rf_center_xs_temp = new();
                List<float> rf_center_ys_temp = new();

                NDArray RF_Center_Xs;
                NDArray RF_Center_Xs_mat = new NDArray();
                NDArray RF_Center_Ys;
                NDArray RF_Center_Ys_mat = new NDArray();

                for (int j = 0; j < scoreMap.Shape[1]; j++) {
                    rf_center_xs_temp.Add(_receptiveFieldCenterStart[i] + _receptiveFieldStride[i] * j);
                }

                RF_Center_Xs = new NDArray(rf_center_xs_temp.ToArray());
                RF_Center_Xs_mat = nd.Tile(RF_Center_Xs, new Shape(scoreMap.Shape[0], 1));

                for (int k = 0; k < scoreMap.Shape[0]; k++) {
                    rf_center_ys_temp.Add(_receptiveFieldCenterStart[i] + _receptiveFieldStride[i] * k);
                }

                RF_Center_Ys = new NDArray(rf_center_ys_temp.ToArray());
                RF_Center_Ys_mat = nd.Tile(RF_Center_Ys, new Shape(scoreMap.Shape[1], 1)).T;

                NDArray x_lt_mat = RF_Center_Xs_mat - bboxMap[0] * constant[i];
                NDArray y_lt_mat = RF_Center_Ys_mat - bboxMap[1] * constant[i];
                NDArray x_rb_mat = RF_Center_Xs_mat - bboxMap[2] * constant[i];
                NDArray y_rb_mat = RF_Center_Ys_mat - bboxMap[3] * constant[i];

                x_lt_mat = x_lt_mat / resizeScale;
                x_lt_mat = x_lt_mat.Clip(0, float.MaxValue);

                y_lt_mat = y_lt_mat / resizeScale;
                y_lt_mat = y_lt_mat.Clip(0, float.MaxValue);

                x_rb_mat = x_rb_mat / resizeScale;
                x_rb_mat = x_rb_mat.Clip(float.MinValue, image.Width);

                y_rb_mat = y_rb_mat / resizeScale;
                y_rb_mat = y_rb_mat.Clip(float.MinValue, image.Height);

                ndarray[] selectIndex = (ndarray[]) np.where(scoreMap.AsNumpy() > scoreThreshold);
                if (selectIndex[0].size == 0)
                    continue;

                for (int l = 0; l < selectIndex[0].Size; l++) {
                    var indexA = (long) selectIndex[0][l];
                    var indexB = (long) selectIndex[1][l];
                    var score = (float) scoreMap.AsNumpy()[indexA, indexB];

                    var bbox = new BoundingBoxParams() {
                        x_lt_mat = (float) x_lt_mat.AsNumpy()[indexA, indexB],
                        y_lt_mat = (float) y_lt_mat.AsNumpy()[indexA, indexB],
                        x_rb_mat = (float) x_rb_mat.AsNumpy()[indexA, indexB],
                        y_rb_mat = (float) y_rb_mat.AsNumpy()[indexA, indexB],
                        scoremap = score
                    };
                    
                    bboxCollection.Add(new Tuple<float, BoundingBoxParams>(score, bbox));
                }

                for (int j = topK; j < 0; j--) {
                    bboxCollection.RemoveAt(topK);
                }
            }
            
            // Create a final list of detected boxes.
            bboxCollection.Sort((firstItem, prevItem) => prevItem.Item1.CompareTo(firstItem.Item1));
            
            List<NDArray> ndArrayList = new();
            foreach (var ent in bboxCollection) {
                ndArrayList.Add(nd.Array(new[]
                    {ent.Item2.x_lt_mat, ent.Item2.y_lt_mat, ent.Item2.x_rb_mat, ent.Item2.y_rb_mat, ent.Item2.scoremap}));
            }

            // Return null if we don't have anything on our bboxcollection.
            if (ndArrayList.Count == 0)
                return null;

            // Finalize our list.
            var stacked = nd.Stack(ndArrayList.ToArray(), ndArrayList.Count);

            #if DEBUG
            Console.WriteLine($"Inference took {s.ElapsedMilliseconds}ms.");
            s.Stop();
            #endif
            
            if (nmsFlag) {
                var nms = NMS(stacked, nmsThreshold);
                return nms;
            }
            return stacked;
        }
        public NDArray NMS(NDArray boxes, float overlapThreshold) {
            #if DEBUG
            Stopwatch s = new Stopwatch();
            s.Start();
            #endif
            
            switch (boxes.Shape[0]) {
                case 0:
                case 1:
                    return boxes;
            }

            if (boxes.DataType != DType.Float32)
                boxes = boxes.AsType(DType.Float32);

            // initialize the list of picked indexes
            List<float> pick = new();
            
            // grab the coordinates of the bounding boxes
            var x1 = boxes.T[0].AsNumpy();
            var x2 = boxes.T[2].AsNumpy();
            var y1 = boxes.T[1].AsNumpy();
            var y2 = boxes.T[3].AsNumpy(); 
            var sc = boxes.T[4].AsNumpy();
            var widths = x2 - x1;
            var heights = y2 - y1;

            // # compute the area of the bounding boxes and sort the bounding
            // # boxes by the bottom-right y-coordinate of the bounding box
            var area = heights * widths;

            var idxs = nd.Argsort(sc);
            List<int> list = new();

            foreach (float item in idxs.AsArray()) {
                list.Add((int)item);
            }

            for (int i = 0; i < list.Count; i++) {
                // # grab the last index in the indexes list and add the
                // # index value to the list of picked indexes
                int last = list.Count - 1;
                int j = list[last];
                pick.Add(j);

                var a1 = np.array(list.ToArray()[..last]);
                
                // # compare second highest score boxes
                var xx1 = np.maximum(x1[j], x1[a1]);
                var yy1 = np.maximum(y1[j], y1[a1]);
                
                var xx2 = np.minimum(x2[j], x2[a1]);
                var yy2 = np.minimum(y2[j], y2[a1]);

                // # compute the width and height of the bounding box
                var w = np.maximum(0, xx2 - xx1 + 1);
                var h = np.maximum(0, yy2 - yy1 + 1);
                
                var overlap = w * h / area[np.array(list.ToArray()[..last])];

                // delete all indexes from the index list that have
                var overlapArr = np.concatenate( np.array(new[]{last}), ((ndarray[])np.where(overlap > overlapThreshold))[0]);
                
                foreach (long indexToRemove in overlapArr) {
                    // Remove the number at that index.
                    list.RemoveAt((int)indexToRemove);
                    
                    // Keep the current index count, or the other indexToRemoves in the current iter will fail.
                    list.Insert((int)indexToRemove, -1);
                }
                
                // Remove the placeholder indexes afterwards.
                list.RemoveAll(x => x == -1);
            }
            
            List<NDArray> finalBoxes = new();

            foreach (var iter in pick) {
                finalBoxes.Add(boxes[iter]);
            }
            
            var retval = nd.Stack(finalBoxes.ToArray(), finalBoxes.Count).Squeeze(1);
            
            #if DEBUG
            s.Stop();
            Console.WriteLine($"NMS took {s.ElapsedMilliseconds}ms.");
            #endif
            
            return retval;
        }

        private struct BoundingBoxParams {
            public float x_lt_mat { get; set; }
            public float y_lt_mat { get; set; }
            public float x_rb_mat { get; set; }
            public float y_rb_mat { get; set; }
            public float scoremap { get; set; }
        }
    }
}