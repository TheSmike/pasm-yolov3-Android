package it.unipr.scarpentim.pasmtftest1;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.test.InstrumentationRegistry;
import android.support.test.runner.AndroidJUnit4;
import android.util.Log;

import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.IOException;
import java.io.InputStream;
import java.util.Iterator;
import java.util.List;

import it.unipr.scarpentim.pasmtftest1.tensorflow.Classifier;
import it.unipr.scarpentim.pasmtftest1.yolo.YoloClassifier;
import it.unipr.scarpentim.pasmtftest1.yolo.YoloV3Classifier;

import static org.junit.Assert.assertEquals;

/**
 * Instrumented test, which will execute on an Android device.
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
@RunWith(AndroidJUnit4.class)
public class Yolov3FromDW2TFTest {

    private static final String TAG = "pasmTEST";
    Context appContext = InstrumentationRegistry.getTargetContext();


//    private static final String YOLO_MODEL_FILE = "file:///android_asset/yolov3-tiny-freeze.bp";
//    private static final String YOLO_INPUT_NAME = "yolov3-tiny/net1";
//    private static final String YOLO_OUTPUT_NAMES = "yolov3-tiny/convolutional10/BiasAdd";

    private static final String YOLO_MODEL_FILE = "file:///android_asset/yolov3_out3.bp";
    private static final String YOLO_INPUT_NAME = "yolov3/net1";
    private static final String YOLO_OUTPUT_NAMES = "yolov3/convolutional59/BiasAdd,yolov3/convolutional67/BiasAdd,yolov3/convolutional75/BiasAdd";

    private static final int YOLO_BLOCK_SIZE = 32;
    private static final int YOLO_INPUT_SIZE = 608;

    public static final String SAMPLE_IMG = "cargo-bike-with-dog-flickr-grrsh.jpg";
//    public static final String SAMPLE_IMG = "monitor.jpg";

    @Test
    public void instatiate(){

        TensorFlowInferenceInterface inferenceInterface = new TensorFlowInferenceInterface(appContext.getAssets(), YOLO_MODEL_FILE);

        Iterator<Operation> operations = inferenceInterface.graph().operations();
        while (operations.hasNext()){
            Operation next = operations.next();
            Log.d(TAG, "operation name: " + next.name());
        }

        Operation in = inferenceInterface.graph().operation(YOLO_INPUT_NAME);
        Operation out = inferenceInterface.graph().operation(YOLO_OUTPUT_NAMES);

        Log.d(TAG, "in: " + in);
        Log.d(TAG, "out: " + out);


    }

    //@Ignore
    @Test
    public void tryIt() throws IOException {
        YoloV3Classifier detector = (YoloV3Classifier) YoloV3Classifier.create(
                appContext.getAssets(),
                YOLO_MODEL_FILE,
                YOLO_INPUT_SIZE,
                YOLO_INPUT_NAME,
                YOLO_OUTPUT_NAMES,
                YOLO_BLOCK_SIZE);

        Bitmap loadedImage = getBitmapFromTestAssets(SAMPLE_IMG);

        Bitmap redimBitmap = Bitmap.createScaledBitmap(loadedImage, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, false);

        Log.i(TAG, "start = " + System.currentTimeMillis());
        List<Classifier.Recognition> recognitions = detector.recognizeImage(redimBitmap);
        Log.i(TAG, " end  = " + System.currentTimeMillis());

        Log.i(TAG, "yolov3 recognitions = " + recognitions);
    }

    public Bitmap getBitmapFromTestAssets(String fileName) throws IOException {
        Context testContext = InstrumentationRegistry.getInstrumentation().getContext();
        AssetManager assetManager = testContext.getAssets();

        InputStream testInput = assetManager.open(fileName);
        Bitmap bitmap = BitmapFactory.decodeStream(testInput);

        return bitmap;
    }
}
