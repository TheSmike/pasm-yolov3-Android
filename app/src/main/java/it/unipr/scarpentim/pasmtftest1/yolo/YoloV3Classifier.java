package it.unipr.scarpentim.pasmtftest1.yolo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

import it.unipr.scarpentim.pasmtftest1.tensorflow.Classifier;

/** An object detector that uses TF and a YOLO model to detect objects. */
public class YoloV3Classifier implements Classifier {


    // Only return this many results with at least this confidence.
    private static final int MAX_RESULTS = 25;

    private static final int NUM_CLASSES = 80;

    private static final int NUM_BOXES_PER_BLOCK = 3 ;

    // TODO(andrewharp): allow loading anchors and classes
    // from files.
    private static final double[] ANCHORS = {
//            1.08, 1.19,
//            3.42, 4.41,
//            6.63, 11.38,
//            9.42, 5.11,
//            16.62, 10.52

            //0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828
//            10,14,  23,27,  37,58,  81,82,  135,169,  344,319 //yolov3 tiny
            10,13,  16,30,  33,23,  30,61,  62,45,  59,119, 116,90,  156,198,  373,326 // yolov3
            // 116,90,  156,198,  373,326 // yolov3 ultimi 3
            //116,90,  156,198,  373,326  // yolov3 primi 3
            //116,90,  156,198,  373,326, 30,61,  62,45,  59,119, 10,13,  16,30,  33,23  // yolov3 reverse
    };

    private static final String[] LABELS = {
            "person",
            "bicycle",
            "car",
            "motorbike",
            "aeroplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "sofa",
            "pottedplant",
            "bed",
            "diningtable",
            "toilet",
            "tvmonitor",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
    };
    private static final String TAG = "pasm-YoloClassifier";

    // Config values.
    private String inputName;
    private int inputSize;

    // Pre-allocated buffers.
    private int[] intValues;
    private float[] floatValues;
    private String[] outputNames;

    private int blockSize;

    private boolean logStats = false;

    private TensorFlowInferenceInterface inferenceInterface;

    /** Initializes a native TensorFlow session for classifying images. */
    public static Classifier create(
            final AssetManager assetManager,
            final String modelFilename,
            final int inputSize,
            final String inputName,
            final String outputName,
            final int blockSize) {
        YoloV3Classifier d = new YoloV3Classifier();
        d.inputName = inputName;
        d.inputSize = inputSize;

        // Pre-allocate buffers.
        d.outputNames = outputName.split(",");
        d.intValues = new int[inputSize * inputSize];
        d.floatValues = new float[inputSize * inputSize * 3];
        d.blockSize = blockSize;

        d.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);

        return d;
    }

    private YoloV3Classifier() {}

    private float expit(final float x) {
        return (float) (1. / (1. + Math.exp(-x)));
    }

    private void softmax(final float[] vals) {
        float max = Float.NEGATIVE_INFINITY;
        for (final float val : vals) {
            max = Math.max(max, val);
        }
        float sum = 0.0f;
        for (int i = 0; i < vals.length; ++i) {
            vals[i] = (float) Math.exp(vals[i] - max);
            sum += vals[i];
        }
        for (int i = 0; i < vals.length; ++i) {
            vals[i] = vals[i] / sum;
        }
    }

    @Override
    public List<Recognition> recognizeImage(final Bitmap bitmap) {

        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int i = 0; i < intValues.length; ++i) {
            floatValues[i * 3 + 0] = ((intValues[i] >> 16) & 0xFF) / 255.0f;
            floatValues[i * 3 + 1] = ((intValues[i] >> 8) & 0xFF) / 255.0f;
            floatValues[i * 3 + 2] = (intValues[i] & 0xFF) / 255.0f;
        }
        Trace.endSection(); // preprocessBitmap

        // Copy the input data into TensorFlow.
        Trace.beginSection("feed");
        inferenceInterface.feed(inputName, floatValues, 1, inputSize, inputSize, 3);
        Trace.endSection();

        // Run the inference call.
        Trace.beginSection("run");
        inferenceInterface.run(outputNames, logStats);
        Trace.endSection();

        // Copy the output Tensor back into the output array.
        Trace.beginSection("fetch");
        int blockSize0 = blockSize;
        int blockSize1 = blockSize/2;
        int blockSize2 = blockSize/4;

        int gridWidth0 = bitmap.getWidth() / blockSize0;
        int gridHeight0 = bitmap.getHeight() / blockSize0;
        int gridWidth1 = bitmap.getWidth() / blockSize1;
        int gridHeight1 = bitmap.getHeight() / blockSize1;
        int gridWidth2 = bitmap.getWidth() / blockSize2;
        int gridHeight2 = bitmap.getHeight() / blockSize2;

        //output dovrebbe essere di dim: 13 * 13 * ( 80 + 5 ) * 3;
        final float[] output0 = new float[gridWidth0 * gridHeight0 * (NUM_CLASSES + 5) * NUM_BOXES_PER_BLOCK];
        Log.i(TAG,  String.format("output0 size is --> %d * %d * (%d + 5) * %d = %d", gridWidth0, gridHeight0, NUM_CLASSES, NUM_BOXES_PER_BLOCK, gridWidth0 * gridHeight0 * (NUM_CLASSES + 5) * NUM_BOXES_PER_BLOCK ));

        final float[] output1 = new float[gridWidth1 * gridHeight1 * (NUM_CLASSES + 5) * NUM_BOXES_PER_BLOCK];
        Log.i(TAG,  String.format("output1 size is --> %d * %d * (%d + 5) * %d = %d", gridWidth1, gridHeight1, NUM_CLASSES, NUM_BOXES_PER_BLOCK, gridWidth1 * gridHeight1 * (NUM_CLASSES + 5) * NUM_BOXES_PER_BLOCK ));

        final float[] output2 = new float[gridWidth2 * gridHeight2 * (NUM_CLASSES + 5) * NUM_BOXES_PER_BLOCK];
        Log.i(TAG,  String.format("output2 size is --> %d * %d * (%d + 5) * %d = %d", gridWidth2, gridHeight2, NUM_CLASSES, NUM_BOXES_PER_BLOCK, gridWidth2 * gridHeight2 * (NUM_CLASSES + 5) * NUM_BOXES_PER_BLOCK ));

        inferenceInterface.fetch(outputNames[0], output0);
        inferenceInterface.fetch(outputNames[1], output1);
        inferenceInterface.fetch(outputNames[2], output2);
        Trace.endSection();

        final ArrayList<Recognition> recognitions = new ArrayList<>();
        populateRecognitions(recognitions, bitmap, output0, gridWidth0, gridHeight0, blockSize0, 0);
        populateRecognitions(recognitions, bitmap, output1, gridWidth1, gridHeight1, blockSize1, 1); //FIXME non funziona
        populateRecognitions(recognitions, bitmap, output2, gridWidth2, gridHeight2, blockSize2, 2); //FIXME non funziona

        Trace.endSection(); // "recognizeImage"

        return recognitions;
    }

    private void populateRecognitions(ArrayList<Recognition> recognitions, Bitmap bitmap, float[] networkOutput, int gridWidth, int gridHeight, int blockSize, int anchorOffset) {
        // Find the best detections.
        final PriorityQueue<Recognition> pq =
                new PriorityQueue<Recognition>(
                        1,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(final Recognition lhs, final Recognition rhs) {
                                // Intentionally reversed to put high confidence at the head of the queue.
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        for (int y = 0; y < gridHeight; ++y) {
            for (int x = 0; x < gridWidth; ++x) {
                for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {
                    final int offset =
                            (gridWidth * (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5))) * y
                                    + (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5)) * x
                                    + (NUM_CLASSES + 5) * b;

//                    final float xPos = (x+1 + expit(networkOutput[offset + 0])) * blockSize;
//                    final float yPos = (y+1 + expit(networkOutput[offset + 1])) * blockSize;

//                    final float w = (float) (Math.exp(output[offset + 2]) * ANCHORS[2 * b + 0]) * blockSize;
//                    final float h = (float) (Math.exp(output[offset + 3]) * ANCHORS[2 * b + 1]) * blockSize;

//                    final float w = (float) (Math.exp(output[offset + 2]) ) * blockSize;
//                    final float h = (float) (Math.exp(output[offset + 3]) ) * blockSize;

//                    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
//                    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;

//                    final float w = (float) (Math.exp(networkOutput[offset + 2]) * ANCHORS[anchorOffset + 2 * b + 0] );
//                    final float h = (float) (Math.exp(networkOutput[offset + 3]) * ANCHORS[anchorOffset + 2 * b + 1] );


                    final float xPos = (x +  networkOutput[offset + 0]) / gridWidth;
                    final float yPos = (y + networkOutput[offset + 1]) / gridHeight;
                    final float w = (float) (Math.exp(networkOutput[offset + 2]) * ANCHORS[anchorOffset + 2 * b + 0] / bitmap.getWidth());
                    final float h = (float) (Math.exp(networkOutput[offset + 3]) * ANCHORS[anchorOffset + 2 * b + 1] / bitmap.getHeight());
					/*
					//https://github.com/pjreddie/darknet/blob/f6d861736038da22c9eb0739dca84003c5a5e275/src/yolo_layer.c
					box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride){
						box b;
						b.x = (i + x[index + 0*stride]) / lw;
						b.y = (j + x[index + 1*stride]) / lh;
						b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
						b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
						return b;
					}
					*/
					
                    final RectF rect =
                            new RectF(
                                    Math.max(0, xPos - w / 2),
                                    Math.max(0, yPos - h / 2),
                                    Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                                    Math.min(bitmap.getHeight() - 1, yPos + h / 2));
                    final float confidence = expit(networkOutput[offset + 4]);

                    int detectedClass = -1;
                    float maxClass = 0;

                    final float[] classes = new float[NUM_CLASSES];
                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        classes[c] = networkOutput[offset + 5 + c];
                    }
                    softmax(classes);

                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        if (classes[c] > maxClass) {
                            detectedClass = c;
                            maxClass = classes[c];
                        }
                    }

                    final float confidenceInClass = maxClass * confidence;
                    if (confidenceInClass > 0.01) {

                        Log.i(TAG, String.format("%s (%d) %f %s", LABELS[detectedClass], detectedClass, confidenceInClass, rect));
                        pq.add(new Recognition("" + offset, LABELS[detectedClass], confidenceInClass, rect));
                    }
                }
            }
        }


        for (int i = 0; i < Math.min(pq.size(), MAX_RESULTS); ++i) {
            recognitions.add(pq.poll());
        }
    }

    @Override
    public void enableStatLogging(final boolean logStats) {
        this.logStats = logStats;
    }

    @Override
    public String getStatString() {
        return inferenceInterface.getStatString();
    }

    @Override
    public void close() {
        inferenceInterface.close();
    }
	
	
	

}