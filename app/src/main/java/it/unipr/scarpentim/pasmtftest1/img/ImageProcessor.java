package it.unipr.scarpentim.pasmtftest1.img;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

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

    public ImageProcessor(Context appContext) {
        this.appContext = appContext;
    }

    public void loadImage(Bitmap loadedImage) {
        loadImage(loadedImage, loadedImage.getWidth(), loadedImage.getHeight());
    }

    public Mat drawBoxes(List<Classifier.Recognition> boxes){
        rgbImage.copyTo(boxesImage);
        Scalar color = new Scalar(0, 125, 0);
        Map<String, Scalar> labels = new HashMap<>();
        Random ran = new Random();

        for (Classifier.Recognition box : boxes) {
            if (!labels.containsKey(box.getTitle())) {

                int r = ran.nextInt(255);
                int g = ran.nextInt(255);
                int b = ran.nextInt(255);

                if (r +g +b < 255/2)
                    g+= 125;


                labels.put(box.getTitle(), new Scalar(r,g,b));
            }
        }

        for (Classifier.Recognition box : boxes) {
            Log.i(TAG, String.valueOf(box));
            if (box.getConfidence() > 0.2) {
//                color.val[0] = (color.val[0] + 25) % 255;
//                color.val[1] = (color.val[1] + 35) % 255;
//                color.val[2] = (color.val[2] + 45) % 255;
                color = labels.get(box.getTitle());

                Point pt1 = new Point(box.getLocation().left * widthRatio, box.getLocation().top * heightRatio);
                Point pt2 = new Point(box.getLocation().right * widthRatio, box.getLocation().bottom * heightRatio);
                Imgproc.rectangle(boxesImage, pt1, pt2, color, 3, 8);
                Point pt3 = new Point(box.getLocation().left * widthRatio, box.getLocation().top * heightRatio);
                Point pt4 = new Point(Math.min(box.getLocation().right, box.getLocation().left + (box.getTitle().length() * 7)) * widthRatio, (box.getLocation().top + 11) * heightRatio);
                Imgproc.rectangle(boxesImage, pt3, pt4, color, FILLED,8);

                pt1.set(new double[] {pt1.x, (pt1.y + 10)});
                Imgproc.putText(boxesImage, box.getTitle(), pt1, Core.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(255,255,255), 2, LINE_AA);
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
