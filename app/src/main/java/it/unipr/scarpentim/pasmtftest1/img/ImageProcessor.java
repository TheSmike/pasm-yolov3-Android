package it.unipr.scarpentim.pasmtftest1.img;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import it.unipr.scarpentim.pasmtftest1.tensorflow.Classifier;

import static org.opencv.core.Core.FILLED;
import static org.opencv.core.Core.LINE_AA;

public class ImageProcessor {

    private static final String TAG = "pasm-YoloClassifier";

    private Mat rgbImage = new Mat();
    private Mat boxesImage = new Mat();
    private Context appContext;
    private float widthRatio;
    private float heightRatio;
    private Map<String, Scalar> colors;

    public ImageProcessor(Context appContext, String[] labels) {
        this.appContext = appContext;

        colors = new HashMap<>();

        int r = 105;
        int g = 0;
        int b = 0;

        for (int i = 0; i < labels.length; i++) {
            r = (r + ((i+0) % 3 == 0 ? 0 : 115)) % 256;
            g = (g + ((i+1) % 3 == 0 ? 0 : 114)) % 256;
            b = (b + ((i+2) % 3 == 0 ? 0 : 113)) % 256;
            colors.put(labels[i], new Scalar(r,g,b));
        }
    }

    public void loadImage(Bitmap loadedImage) {
        loadImage(loadedImage, loadedImage.getWidth(), loadedImage.getHeight());
    }

    public Mat drawBoxes(List<Classifier.Recognition> boxes, double confidenceThreshold){
        rgbImage.copyTo(boxesImage);
        Scalar color;


        for (Classifier.Recognition box : boxes) {
            Log.i(TAG, String.valueOf(box));
            if (box.getConfidence() > confidenceThreshold) {
//                color.val[0] = (color.val[0] + 25) % 255;
//                color.val[1] = (color.val[1] + 35) % 255;
//                color.val[2] = (color.val[2] + 45) % 255;
                color = colors.get(box.getTitle());

                Point pt1 = new Point(box.getLocation().left * widthRatio, box.getLocation().top * heightRatio);
                Point pt2 = new Point(box.getLocation().right * widthRatio, box.getLocation().bottom * heightRatio);
                Imgproc.rectangle(boxesImage, pt1, pt2, color, 3, 8);
                Point pt3 = new Point(box.getLocation().left * widthRatio, box.getLocation().top * heightRatio);
                Point pt4 = new Point(Math.min(box.getLocation().right, box.getLocation().left + (box.getTitle().length() * 7)) * widthRatio, (box.getLocation().top + 11) * heightRatio);
                Imgproc.rectangle(boxesImage, pt3, pt4, color, FILLED,8);

                pt1.set(new double[] {pt1.x + 2*heightRatio, (pt1.y + 10*heightRatio)});
                Imgproc.putText(boxesImage, box.getTitle(), pt1, Core.FONT_HERSHEY_SIMPLEX, 0.4 * heightRatio, new Scalar(255,255,255), (int) (1 * heightRatio), LINE_AA);
            }
        }

        return boxesImage;
    }

    public void loadImage(Bitmap loadedImage, int yoloWidth, int yoloHeight) {
        this.widthRatio = (float)loadedImage.getWidth() / yoloWidth;
        this.heightRatio = (float)loadedImage.getHeight() / yoloHeight;

        Bitmap bmp32 = loadedImage.copy(Bitmap.Config.ARGB_8888, true);
        Utils.bitmapToMat(bmp32, rgbImage);

    }
}
