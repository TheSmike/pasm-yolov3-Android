package it.unipr.scarpentim.pasmtftest1.yolo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Trace;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

import it.unipr.scarpentim.pasmtftest1.tensorflow.Classifier;

/** An object detector that uses TF and a YOLO model to detect objects. */
public class YoloV3ClassifierUltimate implements Classifier {

    // Only return this many results with at least this confidence.
    private static final int MAX_RESULTS = 15;

    private static final int NUM_CLASSES = 80;

    private static final int NUM_BOXES_PER_BLOCK = 3 ;

    private final static float OVERLAP_THRESHOLD = 0.5f;
    public static final String FILE_ANDROID_ASSET = "file:///android_asset/";

    private int[] anchors;
    private String[] labels;

    private static final String TAG = "pasm-YoloV2Classifier";

    // Config values.
    private String inputName;
    private int inputSize;

    // Pre-allocated buffers.
    private int[] intValues;
    private float[] floatValues;
    private String[] outputNames;

    private int[] blockSize;
    private int centerOffset;

    private boolean logStats = false;

    private TensorFlowInferenceInterface inferenceInterface;

    /** Initializes a native TensorFlow session for classifying images. */
    public static Classifier create(
            final AssetManager assetManager,
            final String modelName,
            final int inputSize,
            final String inputName,
            final String outputName,
            final int[] blockSize,
            final int centerOffset) throws IOException {
        YoloV3ClassifierUltimate d = new YoloV3ClassifierUltimate();
        d.inputName = inputName;
        d.inputSize = inputSize;

        // Pre-allocate buffers.
        d.outputNames = outputName.split(",");
        d.intValues = new int[inputSize * inputSize];
        d.floatValues = new float[inputSize * inputSize * 3];
        d.blockSize = blockSize;

        String modelFilename = modelName + ".bp";
        String labelsFilename = modelName + "-labels.txt";
        String anchorsFilename = modelName + "-anchors.txt";

        d.inferenceInterface = new TensorFlowInferenceInterface(assetManager, FILE_ANDROID_ASSET + modelFilename);

        InputStream labelsFile = assetManager.open(labelsFilename);
        InputStream anchorsFile = assetManager.open(anchorsFilename);

        d.labels = streamToLabels(labelsFile);
        d.anchors = streamToAnchors(anchorsFile);

        d.centerOffset = centerOffset;

        return d;
    }

    private YoloV3ClassifierUltimate() {}

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

        final ArrayList<Recognition> recognitions = new ArrayList<>();
        for (int i = 0; i < blockSize.length; i++) {

            // Copy the output Tensor back into the output array.
            Trace.beginSection("fetch i");
            int gridWidth = bitmap.getWidth() / blockSize[i];
            int gridHeight = bitmap.getHeight() / blockSize[i];

            final float[] output = new float[gridWidth * gridHeight * (NUM_CLASSES + 5) * NUM_BOXES_PER_BLOCK];
            Log.d(TAG,  String.format("output0 size is --> %d * %d * (%d + 5) * %d = %d", gridWidth, gridHeight, NUM_CLASSES, NUM_BOXES_PER_BLOCK, gridWidth * gridHeight * (NUM_CLASSES + 5) * NUM_BOXES_PER_BLOCK ));
            inferenceInterface.fetch(outputNames[i], output);
            Trace.endSection();

            populateRecognitions(recognitions, bitmap, output, gridWidth, gridHeight, blockSize[i], i);
        }

        Trace.endSection(); // "recognizeImage"

        return recognitions;
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

    @Override
    public String[] getLabels() {
        return labels;
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

        int y;
        int x;
        for (y = 0; y < gridHeight; ++y) {
            for (x = 0; x < gridWidth; ++x) {
                for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {
                    final int offset =
                            (gridWidth * (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5))) * y
                                    + (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5)) * x
                                    + (NUM_CLASSES + 5) * b;


                    final float xPos = (x + centerOffset + expit(networkOutput[offset + 0])) * blockSize;
                    final float yPos = (y + centerOffset + expit(networkOutput[offset + 1])) * blockSize;

                    final float w = (float) (Math.exp(networkOutput[offset + 2]) * anchors[anchorOffset * 6 + 2 * b + 0]);
                    final float h = (float) (Math.exp(networkOutput[offset + 3]) * anchors[anchorOffset * 6 + 2 * b + 1]);

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
                        classes[c] = networkOutput[offset + 5 + c]; //percentage of each class
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
                        Log.i(TAG, String.format("%s (%d) %f %s", labels[detectedClass], detectedClass, confidenceInClass, rect));
                        pq.add(new Recognition("" + offset, labels[detectedClass], confidenceInClass, rect));
                    }
                }
            }
        }

        getRecognition(recognitions, pq);
    }

    private List<Recognition> getRecognition(ArrayList<Recognition> recognitions, final PriorityQueue<Recognition> priorityQueue) {

        if (priorityQueue.size() > 0) {
            // Best recognition
            Recognition bestRecognition = priorityQueue.poll();
            recognitions.add(bestRecognition);
            int i = 1;
            while(i < MAX_RESULTS) {
                //for (int i = 0; i < Math.min(priorityQueue.size(), MAX_RESULTS); ++i) {
                Recognition recognition = priorityQueue.poll();
                if (recognition == null)
                    break;

                boolean overlaps = false;
                for (Recognition previousRecognition : recognitions) {
                    if (previousRecognition.getTitle().equals( recognition.getTitle())) {
                        overlaps = overlaps || (getIntersectionProportion(previousRecognition.getLocation(),
                                recognition.getLocation()) > OVERLAP_THRESHOLD);
                    }
                }

                if (!overlaps) {
                    recognitions.add(recognition);
                    i++;
                }
            }
        }

        return recognitions;
    }

    private float getIntersectionProportion(RectF primaryShape, RectF secondaryShape) {
        if (overlaps(primaryShape, secondaryShape)) {
            float intersectionSurface = Math.max(0, Math.min(primaryShape.right, secondaryShape.right) - Math.max(primaryShape.left, secondaryShape.left)) *
                    Math.max(0, Math.min(primaryShape.bottom, secondaryShape.bottom) - Math.max(primaryShape.top, secondaryShape.top));

            float surfacePrimary = Math.abs(primaryShape.right - primaryShape.left) * Math.abs(primaryShape.bottom - primaryShape.top);

            return intersectionSurface / surfacePrimary;
        }

        return 0f;
    }

    private boolean overlaps(RectF primary, RectF secondary) {
        return primary.left < secondary.right && primary.right > secondary.left
                && primary.top < secondary.bottom && primary.bottom > secondary.top;
    }

    private static int[] streamToAnchors(InputStream anchorsFile) throws IOException {
        List<Integer> labels = new ArrayList<>();

        // read it with BufferedReader
        BufferedReader br = new BufferedReader(new InputStreamReader(anchorsFile));

        String line;
        while ((line = br.readLine()) != null) {
            String[] split = line.split(",");
            for (String s : split) {
                labels.add(Integer.valueOf(s.trim()));
            }
        }

        br.close();

        int[] ints = new int[labels.size()];
        for (int i = 0; i < labels.size(); i++) {
            ints[i] = labels.get(i);
        }

        return ints;
    }

    private static String[] streamToLabels(InputStream labelsFile) throws IOException {

        List<String> labels = new ArrayList<>();

        // read it with BufferedReader
        BufferedReader br = new BufferedReader(new InputStreamReader(labelsFile));

        String line;
        while ((line = br.readLine()) != null) {
            labels.add(line);
        }

        br.close();

        return labels.toArray(new String[0]);
    }


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

}