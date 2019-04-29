package it.unipr.scarpentim.pasmtftest1;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Environment;
import android.support.test.InstrumentationRegistry;
import android.support.test.runner.AndroidJUnit4;
import android.support.v4.app.ActivityCompat;
import android.util.Log;

import org.apache.commons.lang3.StringUtils;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;

import it.unipr.scarpentim.pasmtftest1.img.ImageProcessor;
import it.unipr.scarpentim.pasmtftest1.tensorflow.Classifier;
import it.unipr.scarpentim.pasmtftest1.yolo.YoloV3Classifier;

/**
 * Instrumented test, which will execute on an Android device.
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
@RunWith(AndroidJUnit4.class)
public class Yolov3DrawRectTest {

    private static final String TAG = "PASM_yolov3";
    Context appContext = InstrumentationRegistry.getTargetContext();

    private static final String YOLO_MODEL_FILE = "yolov3_out3";
    private static final String YOLO_INPUT_NAME = "yolov3/net1";
    private static final String YOLO_OUTPUT_NAMES = "yolov3/convolutional59/BiasAdd,yolov3/convolutional67/BiasAdd,yolov3/convolutional75/BiasAdd";
    private static final int YOLO_INPUT_SIZE = 608;

    private static final int[] YOLO_BLOCK_SIZE = {32, 16, 8};

    public static final String SAMPLE_IMG = "cargo-bike-with-dog-flickr-grrsh.jpg";

    @Test
    public void drawImage() throws IOException {
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, appContext, mLoaderCallback);

        YoloV3Classifier detector = (YoloV3Classifier) YoloV3Classifier.create(
                appContext.getAssets(),
                YOLO_MODEL_FILE,
                YOLO_INPUT_SIZE,
                YOLO_INPUT_NAME,
                YOLO_OUTPUT_NAMES,
                YOLO_BLOCK_SIZE,
                1);

        checkPermissions();
//        Bitmap loadedImage = getBitmapFromTestAssets(SAMPLE_IMG);

        Context testContext = InstrumentationRegistry.getInstrumentation().getContext();
        AssetManager assetManager = testContext.getAssets();

        String pathToImages = "images";
//        String pathToImages = SAMPLE_IMG; // change pathToImages to this for single image

        String[] list = assetManager.list(pathToImages);

        for (int i = 0; i < list.length; i++) {
            if(list[i].endsWith(".jpg")) {
                Bitmap loadedImage = getBitmapFromTestAssets(list[i], pathToImages);

                Bitmap redimBitmap = Bitmap.createScaledBitmap(loadedImage, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, false);

                long start = System.currentTimeMillis();
                List<Classifier.Recognition> recognitions = detector.recognizeImage(redimBitmap);
                long end = System.currentTimeMillis();
                Log.i(TAG, "execution time = " + (end-start));

                Log.i(TAG, "yolov3 recognitions = " + recognitions);

                ImageProcessor processor = new ImageProcessor(testContext, detector.getLabels());
                processor.loadImage(loadedImage, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
                Mat mat = processor.drawBoxes(recognitions, 0.3);
                Mat ultimate = new Mat();
                Imgproc.cvtColor(mat, ultimate, Imgproc.COLOR_RGB2BGR);


                Imgcodecs.imwrite(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM) + "/yolo_v3/boxes" + i + ".jpg", ultimate);
            }
        }

    }

    private void checkPermissions() {
        //check sdcard availability
        Assert.assertEquals("Media is not mounted!", Environment.getExternalStorageState(), Environment.MEDIA_MOUNTED);
        //get and check Permissions
        MyPermissionRequester.request(Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.READ_EXTERNAL_STORAGE);
        int grantResult = ActivityCompat.checkSelfPermission(InstrumentationRegistry.getTargetContext(), Manifest.permission.WRITE_EXTERNAL_STORAGE);
        Assert.assertEquals(PackageManager.PERMISSION_GRANTED, grantResult);
        grantResult = ActivityCompat.checkSelfPermission(InstrumentationRegistry.getContext(), Manifest.permission.WRITE_EXTERNAL_STORAGE);
        Assert.assertEquals(PackageManager.PERMISSION_GRANTED, grantResult);

        //Bitmap ultimateImg = Bitmap.createScaledBitmap(loadedImage, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, false);
        File d = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM), "yolo_v3");
        if (!d.exists()) {
            d.mkdirs();
        }
    }

    private Bitmap getBitmapFromTestAssets(String fileName, String parent) throws IOException {
        Context testContext = InstrumentationRegistry.getInstrumentation().getContext();
        AssetManager assetManager = testContext.getAssets();

        String prefix = "";
        if (StringUtils.isNotEmpty(parent))
            prefix = parent + "/";

        InputStream testInput = assetManager.open(prefix + fileName);
        Bitmap bitmap = BitmapFactory.decodeStream(testInput);

        return bitmap;
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(appContext) {
        @Override
        // Una volta che OpenCV manager è connesso viene chiamato questo metodo di
        public void onManagerConnected(int status) {
            switch (status) {
                // Una volta che OpenCV manager si è connesso con successo
                // possiamo abilitare l'interazione con la tlc
                case LoaderCallbackInterface.SUCCESS:
                    Log.i(TAG, "OpenCV loaded successfully");
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };
}
