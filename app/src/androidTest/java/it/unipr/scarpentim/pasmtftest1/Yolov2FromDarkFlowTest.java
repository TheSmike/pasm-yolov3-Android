package it.unipr.scarpentim.pasmtftest1;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.test.InstrumentationRegistry;
import android.support.test.runner.AndroidJUnit4;
import android.util.Log;

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

import static org.junit.Assert.*;

/**
 * Instrumented test, which will execute on an Android device.
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
@RunWith(AndroidJUnit4.class)
public class Yolov2FromDarkFlowTest {

    private static final String TAG = "pasmTEST";
    Context appContext = InstrumentationRegistry.getTargetContext();
    private static final String MODEL_FILE = "file:///android_asset/yolov2-tiny.pb";

    @Test
    public void useAppContext() {
        // Context of the app under test.
        Context appContext = InstrumentationRegistry.getTargetContext();

        assertEquals("it.unipr.scarpentim.pasmtftest1", appContext.getPackageName());
    }

    @Test
    public void instatiate(){

        TensorFlowInferenceInterface inferenceInterface = new TensorFlowInferenceInterface(appContext.getAssets(), MODEL_FILE);

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

    private static final String YOLO_MODEL_FILE = "file:///android_asset/yolov2-tiny.pb";
    private static final int YOLO_INPUT_SIZE = 416;
    private static final String YOLO_INPUT_NAME = "input";
    private static final String YOLO_OUTPUT_NAMES = "output";
    private static final int YOLO_BLOCK_SIZE = 32;

    @Test
    public void tryIt() throws IOException {
        Classifier detector = YoloClassifier.create(
                appContext.getAssets(),
                YOLO_MODEL_FILE,
                YOLO_INPUT_SIZE,
                YOLO_INPUT_NAME,
                YOLO_OUTPUT_NAMES,
                YOLO_BLOCK_SIZE);

        Bitmap loadedImage = getBitmapFromTestAssets("cargo-bike-with-dog-flickr-grrsh.jpg");

        Bitmap redimBitmap = Bitmap.createScaledBitmap(loadedImage, 416, 416, false);

        List<Classifier.Recognition> recognitions = detector.recognizeImage(redimBitmap);

        Log.i(TAG, "recognitions = " + recognitions);
    }

    public Bitmap getBitmapFromTestAssets(String fileName) throws IOException {
        Context testContext = InstrumentationRegistry.getInstrumentation().getContext();
        AssetManager assetManager = testContext.getAssets();

        InputStream testInput = assetManager.open(fileName);
        Bitmap bitmap = BitmapFactory.decodeStream(testInput);

        return bitmap;
    }
}
